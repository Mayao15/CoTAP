# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def norm(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-5, p=2)

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


class IntraSampleLoss(nn.Module):

    def __init__(self, cfg):
        super(IntraSampleLoss, self).__init__()
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.aux = HuberAuxiliaryLoss(self.cfg.tau)

        # SACL (Feature-SAM) parameters
        self.enable_sacl = getattr(cfg, 'enable_sacl', False)
        self.sacl_rho = getattr(cfg, 'sacl_rho', 0.05)
        self.sacl_gamma = getattr(cfg, 'sacl_gamma', 0.2)
        # Weight for explicit negative similarity penalty to prevent feature collapse
        self.neg_sim_weight = getattr(cfg, 'neg_sim_weight', 1.0)
        # Loss type: 'ap' (CoTAP default) or 'dino' (Standard Contrastive)
        self.loss_type = getattr(cfg, 'loss_type', 'ap')
        
        # DINO parameters
        self.student_temp = getattr(cfg, 'student_temp', 0.1)
        self.teacher_temp = getattr(cfg, 'teacher_temp', 0.07) # Standard DINO teacher temp
        if cfg.loss_type == 'dino':
            self.center_momentum = getattr(cfg, 'center_momentum', 0.9)
            self.nmb_prototypes = cfg.nmb_prototypes if isinstance(cfg.nmb_prototypes, int) else cfg.nmb_prototypes[0]
            
            self.register_buffer("center", torch.zeros(1, self.nmb_prototypes))
            # We need max_steps or max_epochs to schedule teacher temp. 
            # Assuming we can access current epoch or step in forward.
            # Simple linear warmup for teacher temp if needed, or constant.
            # DINO default: warmup from 0.04 to 0.07 over 30 epochs.
            self.teacher_temp_warmup_teacher_temp = getattr(cfg, 'warmup_teacher_temp', 0.04)
            self.teacher_temp_min = getattr(cfg, 'teacher_temp', 0.04) # Using teacher_temp as target
            self.teacher_temp_max = getattr(cfg, 'teacher_temp_max', 0.07) # Or maybe just use fixed?
            # Let's use a simple schedule based on epoch passed in forward
        
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        len_teacher_output = len(teacher_output)
        dist.all_reduce(torch.tensor(len_teacher_output).cuda())
        batch_center = batch_center / len_teacher_output

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def get_teacher_temp(self, epoch, max_epochs=100):
        # Linear warmup for first 30 epochs
        if epoch < 30:
            return self.teacher_temp_warmup_teacher_temp + (self.teacher_temp - self.teacher_temp_warmup_teacher_temp) * epoch / 30
        return self.teacher_temp

    def preprocess_feats(self, feats_gc, nmb_crops):
        # nmb_crops = self.cfg.nmb_crops
        bs = feats_gc.shape[0] // 2 // nmb_crops
        shape = feats_gc.shape[1:]
        perm = [i for i in range(2 + len(shape))]
        perm[:2] = [1, 0]
        
        feats_gc, feats_gc_pos = feats_gc.chunk(2)
        feats_gc = feats_gc.reshape(nmb_crops, bs, *shape).permute(*perm)
        feats_gc_pos = feats_gc_pos.reshape(nmb_crops, bs, *shape).permute(*perm)

        return feats_gc, feats_gc_pos

    def get_img_emb(self, x, keep):
        emb = F.adaptive_avg_pool2d(x * keep, (1,1)).reshape(x.shape[0], -1)
        return emb

    def ranking_ap_cls_loss(self, emb_tea, emb_pos_tea, emb_stu, emb_pos_stu):
        def corr(x, y):
            return torch.einsum('nc,mc->nm', norm(x, -1), norm(y, -1))

        bs, n_gc, d = emb_tea.shape
        sim_tea = corr(emb_tea.reshape(-1, d), emb_pos_tea.reshape(-1, d))
        sim_stu = corr(emb_stu.reshape(-1, d), emb_pos_stu.reshape(-1, d))
        is_match = torch.block_diag(*[torch.ones(n_gc, n_gc, device=emb_tea.device) for _ in range(bs)])

        loss = self.helper(sim_tea.data, sim_stu, is_match)
        return loss

    def ranking_cos_patch_loss(self, ori_feat_tea, ori_feat_pos_tea, \
        ori_feat_stu, ori_feat_pos_stu, ori_keep, return_entropy=False, current_epoch=0):
        # perm = super_perm(len(ori_feat_tea), ori_feat_tea.device)

        bs, n_gc, d, h, w = ori_feat_stu.shape
        ori_keep, ori_keep_pos = ori_keep.reshape(2*bs*n_gc, 1, h, w).chunk(2)
        loss = 0
        total_entropy = 0.0

        for ks in self.cfg.pool_ks:
            feat_stu = F.adaptive_avg_pool2d(ori_feat_stu.reshape(bs*n_gc,d,h,w), (ks,ks))
            feat_tea = F.adaptive_avg_pool2d(ori_feat_tea.reshape(bs*n_gc,d,h,w), (ks,ks))
            feat_pos_stu = F.adaptive_avg_pool2d(ori_feat_pos_stu.reshape(bs*n_gc,d,h,w), (ks,ks))
            feat_pos_tea = F.adaptive_avg_pool2d(ori_feat_pos_tea.reshape(bs*n_gc,d,h,w), (ks,ks))
            keep = F.adaptive_avg_pool2d(ori_keep.reshape(bs*n_gc,1,h,w), (ks,ks))
            keep_pos = F.adaptive_avg_pool2d(ori_keep_pos.reshape(bs*n_gc,1,h,w), (ks,ks))

            feat_stu = feat_stu.permute(0,2,3,1).reshape(bs, n_gc*ks**2, d)
            feat_tea = feat_tea.permute(0,2,3,1).reshape(bs, n_gc*ks**2, d)
            feat_pos_stu = feat_pos_stu.permute(0,2,3,1).reshape(bs, n_gc*ks**2, d)
            feat_pos_tea = feat_pos_tea.permute(0,2,3,1).reshape(bs, n_gc*ks**2, d)
            keep = keep.reshape(bs, n_gc*ks**2)
            keep_pos = keep_pos.reshape(bs, n_gc*ks**2)

            _, idx_keep = torch.topk(keep, k=min(self.cfg.topk, n_gc*ks**2), dim=1)
            _, idx_keep_pos = torch.topk(keep_pos, k=min(self.cfg.topk, n_gc*ks**2), dim=1)
            idx_keep = idx_keep.unsqueeze(2).repeat(1,1,d)
            idx_keep_pos = idx_keep_pos.unsqueeze(2).repeat(1,1,d)
            feat_stu = torch.gather(feat_stu, 1, idx_keep) # [BS, N_keep, D]
            feat_tea = torch.gather(feat_tea, 1, idx_keep)
            feat_pos_stu = torch.gather(feat_pos_stu, 1, idx_keep_pos)
            feat_pos_tea = torch.gather(feat_pos_tea, 1, idx_keep_pos)

            # DINO Loss Logic
            if self.loss_type == 'dino':
                # Flatten to [N_total, D]
                # feat_stu (View 1) <-> feat_pos_tea (View 2)
                # feat_pos_stu (View 2) <-> feat_tea (View 1)
                
                # Student Output
                s1 = feat_stu.reshape(-1, d) / self.student_temp
                s2 = feat_pos_stu.reshape(-1, d) / self.student_temp
                
                # Teacher Output
                # Apply Centering and Sharpening
                temp = self.get_teacher_temp(current_epoch)
                t1 = F.softmax((feat_tea.reshape(-1, d) - self.center) / temp, dim=-1).detach()
                t2 = F.softmax((feat_pos_tea.reshape(-1, d) - self.center) / temp, dim=-1).detach()
                
                # Calculate Loss
                # CE(T2, S1) + CE(T1, S2)
                loss1 = torch.sum(-t2 * F.log_softmax(s1, dim=-1), dim=-1)
                loss2 = torch.sum(-t1 * F.log_softmax(s2, dim=-1), dim=-1)
                
                # Calculate Entropy for SACL weighting or logging
                # Entropy of Teacher Distribution
                entropy1 = -torch.sum(t1 * torch.log(t1 + 1e-8), dim=-1)
                entropy2 = -torch.sum(t2 * torch.log(t2 + 1e-8), dim=-1)
                entropy = (entropy1 + entropy2) / 2
                
                if return_entropy:
                    total_entropy += entropy.mean()
                
                # SACL Weighting
                if self.enable_sacl:
                    entropy_weight = torch.exp(-self.sacl_gamma * entropy)
                    loss1 = loss1 * entropy_weight # Simplified: apply avg weight or per-sample? Per-sample.
                    # Wait, entropy1 corresponds to t1 (used in loss2), entropy2 corresponds to t2 (used in loss1)
                    # Correct mapping:
                    # loss1 uses t2 -> weight by entropy2
                    # loss2 uses t1 -> weight by entropy1
                    
                    w1 = torch.exp(-self.sacl_gamma * entropy2)
                    w2 = torch.exp(-self.sacl_gamma * entropy1)
                    loss1 = loss1 * w1
                    loss2 = loss2 * w2

                loss += (loss1.mean() + loss2.mean()) / 2
                
                # Update Center
                # We update center based on all teacher outputs in this batch step
                # Note: This might be called multiple times per step if multiple pool_ks.
                # Standard DINO updates center once per step based on CLS token.
                # Here we update based on dense features. 
                # Ideally, we should only update once? Or update with average?
                # Let's accumulate or just update. Since it's EMA, frequent updates are fine but might be slow.
                # To match exactly, we should cat all teacher outputs.
                
                self.update_center(torch.cat([t1, t2])) # Should we update with softmaxed output? 
                # DINO update_center uses teacher_output (logits) directly, NOT softmaxed.
                # "teacher_output" in snippet is raw output.
                # "teacher_out = F.softmax((teacher_output - self.center) / temp)"
                # "self.update_center(teacher_output)"
                
                # Correct: Pass raw logits
                raw_t1 = feat_tea.reshape(-1, d)
                raw_t2 = feat_pos_tea.reshape(-1, d)
                self.update_center(torch.cat([raw_t1, raw_t2]))

                continue # Skip the rest of CoTAP logic

            cos_sim_stu_pos = torch.einsum('nxc,nyc->nxy', norm(feat_stu, -1), norm(feat_pos_stu, -1))
            cos_sim_tea_pos = torch.einsum('nxc,nyc->nxy', norm(feat_tea, -1), norm(feat_pos_tea, -1))

            cos_sim_stu_neg, cos_sim_tea_neg = [], []
            for i in range(5):
                perm = super_perm(len(feat_tea), feat_tea.device)
                _cos_sim_stu_neg = torch.einsum('nxc,nyc->nxy', norm(feat_stu, -1), norm(feat_pos_stu[perm], -1))
                _cos_sim_tea_neg = torch.einsum('nxc,nyc->nxy', norm(feat_tea, -1), norm(feat_pos_tea[perm], -1))
                cos_sim_stu_neg.append(_cos_sim_stu_neg)
                cos_sim_tea_neg.append(_cos_sim_tea_neg)
            cos_sim_stu_neg = torch.cat(cos_sim_stu_neg, dim=1)
            cos_sim_tea_neg = torch.cat(cos_sim_tea_neg, dim=1)

            sim_stu = torch.cat([cos_sim_stu_pos, cos_sim_stu_neg], dim=1)
            sim_tea = torch.cat([cos_sim_tea_pos, cos_sim_tea_neg], dim=1)
            is_match = torch.ones(*sim_stu.shape, device=feat_stu.device)
            is_match[:, -cos_sim_stu_neg.shape[1]:] = 0

            # print('--------- ks: %d ----------'%ks)
            # print_tensor(sim_stu, 'sim_stu')
            # print_tensor(sim_tea, 'sim_tea')

            # loss += self.helper(sim_tea, sim_stu, is_match)

            # Entropy Weighting for SACL
            entropy_weight = None
            
            # Always compute entropy if requested or SACL enabled
            if self.enable_sacl or return_entropy:
                # Compute entropy of the teacher distribution
                # sim_tea is [B, N_total]. 
                # Convert cosine sim to probs for entropy calculation.
                tau = getattr(self.cfg, 'tau', 0.1)
                probs = F.softmax(sim_tea / tau, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1) # [B]
                
                if return_entropy:
                    total_entropy += entropy.mean()

                if self.enable_sacl:
                    # Weight: exp(-gamma * H)
                    entropy_weight = torch.exp(-self.sacl_gamma * entropy) # [B]
                    # Expand weight to match flattened size in helper
                    # helper receives flattened tensors. 
                    # We need to pass this weight to helper.
            
            loss += self.helper(sim_tea, sim_stu, is_match, entropy_weight=entropy_weight)

            # Explicit Negative Similarity Penalty
            # To prevent feature collapse (high similarity everywhere), we explicitly penalize
            # the average cosine similarity of negative pairs.
            if self.neg_sim_weight > 0:
                loss += self.neg_sim_weight * cos_sim_stu_neg.mean()

            # sim_stu_all.append(sim_stu.view(1, -1))
            # sim_tea_all.append(sim_tea.view(1, -1))
            # is_match_all.append(is_match.view(1, -1))

        # loss = self.helper(
        #     torch.cat(sim_tea_all, dim=1),
        #     torch.cat(sim_stu_all, dim=1),
        #     torch.cat(is_match_all, dim=1)
        # )
        # return loss.mean()
        
        final_loss = loss / len(self.cfg.pool_ks)
        if return_entropy:
            avg_entropy = total_entropy / len(self.cfg.pool_ks)
            return final_loss, avg_entropy
        else:
            return final_loss

    def ranking_ce_cls_loss(self, emb_tea, emb_pos_tea, emb_stu, emb_pos_stu):
        p = torch.cat([emb_pos_tea, emb_tea])
        log_q = torch.log_softmax(torch.cat([emb_stu, emb_pos_stu]) / self.cfg.temperature_cls, dim=2)
        bs, n_gc, d = p.shape
        loss = -(p.reshape(bs, n_gc, 1, d) * log_q.reshape(bs, 1, n_gc, d)).sum(-1)
        return loss.mean()

    def weight_fn(self, x, pseudo_label=None, alpha=0.1):
        thres = self.cfg.weight_thres
        weight = torch.clamp(x - thres, 0)

        if pseudo_label is not None:
            mask_pos = (pseudo_label == 1).float()
            weight = (1 - mask_pos) * alpha * weight + mask_pos * weight

        return weight

    # def helper(self, targets, preds, pseudo_label=None, **kwargs):
    def helper(self, targets, preds, pseudo_label=None, entropy_weight=None, **kwargs):

        entropy_weight = kwargs.get('entropy_weight', None)
        # Expand entropy_weight if provided to match flattened dimension
        if entropy_weight is not None:
             # entropy_weight is [B]
             # targets is [B, N]
             # We want weight to be [B, N] (same for all N) then flattened to [1, B*N]
             b, n = targets.shape
             expanded_weight = entropy_weight.unsqueeze(1).expand(b, n).reshape(1, -1)

        assert not targets.requires_grad and preds.requires_grad

        targets = targets.view(1, -1)
        preds = preds.view(1, -1)
        pseudo_label = pseudo_label.view(1, -1)

        with torch.no_grad():
            weight = self.weight_fn(targets, pseudo_label)
            if entropy_weight is not None:
                weight = weight * expanded_weight
         
            loss_pp = self.aux.run_pp(targets, preds)

        loss_pn = self.aux.run_pn(targets, preds)
        loss_pn *= self.cfg.ap_mul

        loss = loss_pn / (1e-5 + loss_pp)
        g_loss = (loss / (1 + loss)) * weight
        g_loss = g_loss.sum(-1) / (1e-5 + weight.sum(-1))

        return g_loss.mean()

    def forward_patch(self, outputs_gc_stu, outputs_gc_tea, nmb_crops, masks=None, \
        pl_module=None, prefix='', head_idx=0, **kwargs):

        gc_feat_stu = outputs_gc_stu['feat_%d'%head_idx]
        gc_feat_tea = outputs_gc_tea['feat_%d'%head_idx]
        attn = outputs_gc_tea['attn']
        n_gc = nmb_crops
        bs = gc_feat_stu.shape[0] // n_gc
        d, h_gc, w_gc = gc_feat_stu.shape[1:]

        _keep = torch.ones(bs, n_gc, h_gc, w_gc, device=gc_feat_stu.device)
        if attn is not None:
            _keep *= attn.view(bs, n_gc, h_gc, w_gc).float()

        gc_feat_stu, gc_feat_pos_stu = self.preprocess_feats(gc_feat_stu, n_gc)
        gc_feat_tea, gc_feat_pos_tea = self.preprocess_feats(gc_feat_tea, n_gc)

        # Standard Loss Calculation or SACL
        if self.enable_sacl:
            # 1. Compute Gradients for Feature-SAM
            # We need to enable gradients for the student features to calculate the perturbation
            # Note: gc_feat_stu and gc_feat_pos_stu are derived from outputs_gc_stu
            # We should detach them to avoid double backprop issues if we were optimizing the model here,
            # but we just want gradients w.r.t these features.
            # However, since we want to modify the graph input for the final loss, 
            # let's work on a detached clone for gradient calculation.
            
            feat_stu_grad = gc_feat_stu.detach().clone().requires_grad_(True)
            feat_pos_stu_grad = gc_feat_pos_stu.detach().clone().requires_grad_(True)

            loss_for_grad = self.ranking_cos_patch_loss(
                gc_feat_tea, gc_feat_pos_tea,
                feat_stu_grad, feat_pos_stu_grad, _keep,
                current_epoch=pl_module.current_epoch if pl_module else 0
            )
            
            # 2. Generate Adversarial Perturbation
            # We want to MAXIMIZE loss, so we move in direction of gradient.
            # Gradients w.r.t features
            grads = torch.autograd.grad(loss_for_grad, [feat_stu_grad, feat_pos_stu_grad])
            
            # Normalize gradients (L2 norm per feature vector)
            # Shapes are [nmb_crops, bs, D, H, W]
            # We normalize over D dimension (dim=2)
            grad_stu, grad_pos_stu = grads
            
            def compute_delta(grad, rho):
                norm = torch.norm(grad, p=2, dim=2, keepdim=True) + 1e-8
                return rho * grad / norm

            delta_stu = compute_delta(grad_stu, self.sacl_rho)
            delta_pos_stu = compute_delta(grad_pos_stu, self.sacl_rho)
            
            # 3. Compute Final Loss with Perturbed Features
            # We treat perturbation as constant (detach)
            perturbed_feat_stu = gc_feat_stu + delta_stu.detach()
            perturbed_feat_pos_stu = gc_feat_pos_stu + delta_pos_stu.detach()
            
            loss_patch, mean_entropy = self.ranking_cos_patch_loss(
                gc_feat_tea, gc_feat_pos_tea,
                perturbed_feat_stu, perturbed_feat_pos_stu, _keep,
                return_entropy=True,
                current_epoch=pl_module.current_epoch if pl_module else 0
            )
            
        else:
            loss_patch, mean_entropy = self.ranking_cos_patch_loss(
                gc_feat_tea, gc_feat_pos_tea,
                gc_feat_stu, gc_feat_pos_stu, _keep,
                return_entropy=True,
                current_epoch=pl_module.current_epoch if pl_module else 0
            )

        losses = [
            {
                'name': 'loss_%s_patch_intra'%prefix,
                'loss': loss_patch.mean(),
                'weight': self.cfg.weight_patch_intra
            },
            {
                'name': 'entropy_%s_patch_intra'%prefix,
                'loss': mean_entropy,
                'weight': 0 # Just for logging
            }
        ]

        return losses

    def forward_cls(self, outputs_gc_stu, outputs_gc_tea, nmb_crops, \
        pl_module=None, prefix='', head_idx=0, **kwargs):

        if not 'cls_%d_sk'%head_idx in outputs_gc_tea.keys():
            return [{
                'name': 'loss_%s_cls_intra'%prefix,
                'loss': 0 * outputs_gc_stu['cls_%d'%head_idx].mean(),
                'weight': self.cfg.weight_cls_intra
            }]

        gc_emb_stu = outputs_gc_stu['cls_%d'%head_idx]
        gc_emb_tea_sk = outputs_gc_tea['cls_%d_sk'%head_idx]

        if len(gc_emb_stu) != len(gc_emb_tea_sk):        
            bs, d = gc_emb_tea_sk.shape
            bs = bs // 2 // self.cfg.nmb_crops[0]
            gc_emb_tea_sk = gc_emb_tea_sk.view(bs, 2, self.cfg.nmb_crops[0], d)
            gc_emb_tea_sk = gc_emb_tea_sk.repeat(1, self.cfg.nmb_crops[1]//self.cfg.nmb_crops[0], 1, 1).reshape(-1, d)

        gc_emb_stu, gc_emb_pos_stu = self.preprocess_feats(gc_emb_stu, nmb_crops)
        gc_emb_tea_sk, gc_emb_pos_tea_sk = self.preprocess_feats(gc_emb_tea_sk, nmb_crops)
        loss_cls = self.ranking_ce_cls_loss(gc_emb_tea_sk, gc_emb_pos_tea_sk, \
            gc_emb_stu, gc_emb_pos_stu)

        losses = [
            {
                'name': 'loss_%s_cls_intra'%prefix,
                'loss': loss_cls.mean(),
                'weight': self.cfg.weight_cls_intra
            }
        ]

        return losses

    def forward(self, outputs_stu, outputs_tea, lc_outputs_stu, lc_outputs_tea, masks, pl_module, **kwargs):
        args = {
            'outputs_gc_stu': outputs_stu,
            'outputs_gc_tea': outputs_tea,
            'nmb_crops': self.cfg.nmb_crops[0],
            'masks': masks,
            'pl_module': pl_module,
            'prefix': self.cfg.prefix,
            'eps': 1e-5
        }
        losses = []
        if self.cfg.weight_patch_intra > 0:
            losses += self.forward_patch(head_idx=self.cfg.head_idx_patch, **args)
        if self.cfg.weight_cls_intra > 0:
            losses += self.forward_cls(head_idx=self.cfg.head_idx_cls, **args)

        # args = {
        #     'outputs_gc_stu': lc_outputs_stu,
        #     'outputs_gc_tea': outputs_tea,
        #     'nmb_crops': self.cfg.nmb_crops[1],
        #     'masks': masks,
        #     'pl_module': pl_module,
        #     'prefix': 'lc_' + self.cfg.prefix,
        #     'eps': 1e-5
        # }

        # if self.cfg.weight_cls_intra_local > 0:
        #     losses += self.forward_cls(head_idx=self.cfg.head_idx_cls, **args)

        return losses


class HuberAuxiliaryLoss(object):
    def __init__(self, tau):
        self.tau = tau

    def run_pn(self, targets, preds):
        tau = self.tau

        def _surrogate_fn_pn(x):
            x = torch.clamp(x, max=tau)
            return torch.where(
                x >= 0,
                (x / tau - 1)*(x / tau - 1),
                (-2 * x / tau + 1)
            )

        def _surrogate_target(x):
            return torch.clamp(x, 0)

        b, n = targets.shape
        diff_target = _surrogate_target(targets.view(b,n,1) - targets.view(b,1,n))
        diff_pred = _surrogate_fn_pn(preds.view(b,n,1) - preds.view(b,1,n))

        # mask_ignore = (pseudo_label == -1).float()
        # mask = (1 - mask_ignore).view(b,n,1) * (1 - mask_ignore).view(b,1,n)

        # return (diff_target * diff_pred * mask).sum(-1)
        return (diff_target * diff_pred).sum(-1)

    def run_pp(self, targets, preds):

        def _surrogate_fn_pp(x):
            return (x <= 0).float()

        def _surrogate_target(x):
            return (x <= 0).float()

        b, n = preds.shape
        diff_target = _surrogate_target(targets.view(b,n,1) - targets.view(b,1,n))
        diff_pred = _surrogate_fn_pp(preds.view(b,n,1) - preds.view(b,1,n))

        # mask_ignore = (pseudo_label == -1).float()
        # mask = (1 - mask_ignore).view(b,n,1) * (1 - mask_ignore).view(b,1,n)

        # return (diff_target * diff_pred * mask).sum(-1)
        return (diff_target * diff_pred).sum(-1)


class HingeAuxiliaryLoss(object):
    def __init__(self, tau):
        self.tau = tau

    def run_pn(self, targets, preds):
        tau = self.tau
        targets = targets.float()
        preds = preds.float()

        def _surrogate_fn_pn(x):
            return torch.clamp(tau - x, 0)

        def _surrogate_fn_pn_acc(x_pos, x_neg, y_pos, y_neg):

            def get_suf_sum(x_tensor):
                pre_sum = torch.cumsum(x_tensor, 1)
                pre_sum = torch.cat([torch.zeros(len(pre_sum), 1, device=x_tensor.device), pre_sum], 1)
                suf_sum = pre_sum[:, -1:] - pre_sum
                return suf_sum

            sort_idx = torch.sort(x_neg, 1)[1]
            x_neg = torch.gather(x_neg, 1, sort_idx)
            y_neg = torch.gather(y_neg, 1, sort_idx)
            idx = torch.searchsorted(x_neg, x_pos - tau)

            suf_sum_x_neg = get_suf_sum(x_neg)
            suf_sum_y_neg = get_suf_sum(y_neg)
            suf_sum_xy_neg = get_suf_sum(x_neg * y_neg)

            loss = 0
            loss += y_pos * torch.gather(suf_sum_x_neg, 1, idx)
            loss += (x_pos - tau) * torch.gather(suf_sum_y_neg, 1, idx)
            loss -= torch.gather(suf_sum_xy_neg, 1, idx)
            loss += y_pos * (tau - x_pos) * (y_neg.shape[1] - idx)

            return loss

        def _solve_pn_acc(targets, preds, ans):
            if targets.shape[1] <= 1000:
                n = targets.shape[0]
                loss = _surrogate_fn_pn(preds.view(n, -1, 1) - preds.view(n, 1, -1))
                ans += (loss * torch.clamp(targets.view(n, -1, 1) - targets.view(n, 1, -1), 0).float()).sum(-1)
                # ans += (loss * (targets.view(n, -1, 1) > targets.view(n, 1, -1)).float()).sum(-1)
                return

            mid = preds.shape[1] // 2
            ans[:, :mid] += _surrogate_fn_pn_acc(
                preds[:, :mid], 
                preds[:, mid:], 
                targets[:, :mid],
                targets[:, mid:]
            )
            _solve_pn_acc(targets[:, :mid], preds[:, :mid], ans[:, :mid])
            _solve_pn_acc(targets[:, mid:], preds[:, mid:], ans[:, mid:])

        def solve_pn_acc(targets, preds):
            ans = 0 * preds
            idx = torch.sort(-targets, dim=1)[1]
            targets = torch.gather(targets, 1, idx)
            preds = torch.gather(preds, 1, idx)

            _solve_pn_acc(targets, preds, ans)
            ans = torch.gather(ans, 1, torch.sort(idx, dim=1)[1])

            return ans / tau

        return solve_pn_acc(targets, preds)

    def run_pp(self, targets, preds):        
        targets = targets.float()
        preds = preds.float()

        def _surrogate_fn_pp(x):
            return (x <= 0).float()

        def _surrogate_fn_pp_acc(x, y):
            y = torch.sort(y, 1)[0]
            idx = torch.searchsorted(y, x)
            loss = (y.shape[1] - idx)
            return loss

        def _solve_pp_acc(targets, preds, ans):
            if targets.shape[1] <= 1000:
                n = targets.shape[0]
                loss = _surrogate_fn_pp(preds.view(n, -1, 1) - preds.view(n, 1, -1))
                ans += (loss * (targets.view(n, -1, 1) <= targets.view(n, 1, -1)).float()).sum(-1)
                return

            mid = preds.shape[1] // 2
            ans[:, :mid] += _surrogate_fn_pp_acc(preds[:, :mid], preds[:, mid:])
            _solve_pp_acc(targets[:, :mid], preds[:, :mid], ans[:, :mid])
            _solve_pp_acc(targets[:, mid:], preds[:, mid:], ans[:, mid:])

        def solve_pp_acc(targets, preds):
            ans = torch.zeros_like(preds, device=preds.device)
            idx = torch.sort(targets, dim=1)[1]
            targets = torch.gather(targets, 1, idx)
            preds = torch.gather(preds, 1, idx)
            _solve_pp_acc(targets, preds, ans)

            ans = torch.gather(ans, 1, torch.sort(idx, dim=1)[1])
            return ans

        return solve_pp_acc(targets, preds)


def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
