# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .identity import MemoryBankPatch, MemoryBankCls


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
        self.patch_cluster_queue = MemoryBankPatch(cfg)
        self.cls_cluster_queue = MemoryBankCls(cfg)
        self.aux = HuberAuxiliaryLoss(self.cfg.tau)

    def preprocess_feats(self, feats_gc):
        nmb_crops = self.cfg.nmb_crops
        bs = feats_gc.shape[0] // 2 // nmb_crops[0]
        shape_gc = feats_gc.shape[1:]
        
        feats_gc, feats_gc_pos = feats_gc.chunk(2)
        feats_gc = feats_gc.reshape(nmb_crops[0], bs, *shape_gc).permute(1,0,2)
        feats_gc_pos = feats_gc_pos.reshape(nmb_crops[0], bs, *shape_gc).permute(1,0,2)

        return feats_gc, feats_gc_pos

    def get_img_emb(self, x, attn):
        emb = F.adaptive_avg_pool2d(x * attn, (1,1)).reshape(x.shape[0], -1)
        return emb

    def ranking_loss(self, emb_tea, emb_pos_tea, emb_stu, emb_pos_stu):
        def corr(x, y):
            return torch.einsum('nc,mc->nm', norm(x, -1), norm(y, -1))

        bs, n_gc, d = emb_tea.shape
        sim_tea = corr(emb_tea.reshape(-1, d), emb_pos_tea.reshape(-1, d)).reshape(-1) # (n_gc * bs) * (n_gc * bs)
        sim_stu = corr(emb_stu.reshape(-1, d), emb_pos_stu.reshape(-1, d)).reshape(-1)
        is_match = torch.block_diag(*[torch.ones(n_gc, n_gc, device=emb_tea.device) for _ in range(bs)])
        
        loss = self.helper(sim_tea.data, sim_stu, is_match.view(-1))
        return loss
        
        # sim_tea_pos = torch.index_select(sim_tea, 0, is_match.nonzero().reshape(-1))
        # sim_stu_pos = torch.index_select(sim_stu, 0, is_match.nonzero().reshape(-1))
        # sim_tea_neg = torch.index_select(sim_tea, 0, (1-is_match).nonzero().reshape(-1))
        # sim_stu_neg = torch.index_select(sim_stu, 0, (1-is_match).nonzero().reshape(-1))

        # try:
        #     self.cnt += 1
        # except:
        #     self.cnt = 0
        # if self.cnt % 50 == 0:
        #     print('-'*30)
        #     print_tensor(sim_pos_stu, 'sim_pos_stu')
        #     print_tensor(sim_neg_stu, 'sim_neg_stu')
        #     print_tensor(sim_pos_tea, 'sim_pos_tea')
        #     print_tensor(sim_neg_tea, 'sim_neg_tea')

        # margin = self.cfg.margin
        # diff_tea = sim_tea_pos.reshape(-1, 1) \
        #     - sim_tea_neg.reshape(1, -1)
        # diff_stu = sim_stu_pos.reshape(-1, 1) \
        #     - sim_stu_neg.reshape(1, -1)
        # diff_tea += margin
        # loss = self._surrogate_fn_pn(diff_stu) * torch.clamp(diff_tea, 0)
        # return loss
    
    def weight_fn(self, x):
        thres = self.cfg.weight_thres
        return torch.clamp(x - thres, 0)

    def helper(self, targets, preds, pseudo_label=None, alpha=0.1, **kwargs):

        assert not targets.requires_grad and preds.requires_grad

        targets = targets.view(1, -1)
        preds = preds.view(1, -1)
        pseudo_label = pseudo_label.view(1, -1)

        with torch.no_grad():
            weight = self.weight_fn(targets)
            mask_pos = (pseudo_label == 1).float()
            weight = (1 - mask_pos) * alpha * weight + mask_pos * weight

        loss_pn = self.aux.run_pn(targets, preds, pseudo_label)
        with torch.no_grad():
            loss_pp = self.aux.run_pp(targets, preds, pseudo_label)

        loss = loss_pn / (1e-5 + loss_pp)
        g_loss = (loss / (1 + loss)) * weight
        g_loss = g_loss.sum(-1) / (1e-5 + weight.sum(-1))
        g_loss = g_loss.mean()
        
        return g_loss

    def forward_patch(self, outputs_gc_stu, outputs_gc_tea, masks=None, pl_module=None, prefix='', head_idx=0, **kwargs):
        gc_feat_stu = outputs_gc_stu['feat_%d'%head_idx]
        gc_feat_tea = outputs_gc_tea['feat_%d'%head_idx]
        attn = outputs_gc_tea['attn']
        
        if not self.patch_cluster_queue.ready():
            self.patch_cluster_queue.update(gc_feat_tea, attn, pl_module)
            return [{
                'name': 'loss_%s_patch_inner'%prefix,
                'loss': 0 * gc_feat_stu.mean(),
                'weight': self.cfg.weight_patch_intra
            }]
        
        n_gc = self.cfg.nmb_crops[0]
        bs = gc_feat_stu.shape[0] // n_gc
        d, h_gc, w_gc = gc_feat_stu.shape[1:]

        patch_queue = self.patch_cluster_queue.queue_cluster
        self.patch_cluster_queue.update(gc_feat_tea, attn, pl_module)
        input_sinkhorn = gc_feat_tea.permute(0,2,3,1).reshape(-1, d)
        num_patch_vectors_for_pred = input_sinkhorn.shape[0]
        input_sinkhorn = torch.cat([patch_queue, input_sinkhorn], dim=0)
        patch_feat_tea_sk = self.sinkhorn(input_sinkhorn, pl_module=pl_module)[-num_patch_vectors_for_pred:]
        gc_feat_tea = patch_feat_tea_sk.view(n_gc*bs, h_gc, w_gc, d).permute(0,3,1,2)

        gc_emb_stu = self.get_img_emb(gc_feat_stu, attn)
        gc_emb_tea = self.get_img_emb(gc_feat_tea, attn)
        gc_emb_stu, gc_emb_pos_stu = self.preprocess_feats(gc_emb_stu)
        gc_emb_tea, gc_emb_pos_tea = self.preprocess_feats(gc_emb_tea)

        loss_patch = self.ranking_loss(gc_emb_tea, gc_emb_pos_tea, \
            gc_emb_stu, gc_emb_pos_stu)

        losses = [
            {
                'name': 'loss_%s_patch_intra'%prefix,
                'loss': loss_patch.mean(),
                'weight': self.cfg.weight_patch_intra
            }
        ]

        return losses
    
    def forward(self, outputs_stu, outputs_tea, masks, \
                lc_outputs_stu, bboxes, pl_module, **kwargs):
        args = {
            'outputs_gc_stu': outputs_stu,
            'outputs_gc_tea': outputs_tea,
            'outputs_lc_stu': lc_outputs_stu,
            'masks': masks,
            'bbox_dict': bboxes,
            'pl_module': pl_module,
            'prefix': self.cfg.prefix,
            'eps': 1e-5
        }
        losses = self.forward_patch(**args)
        return losses

    def sinkhorn(self, init_q, nmb_iters=3, pl_module=None) -> torch.Tensor:
        device=init_q.device
        with torch.no_grad():
            Q = torch.exp(init_q.detach() / 0.05).t()

            sum_Q = torch.sum(Q)
            pl_module.reduce(sum_Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0], device=device)
            r = torch.ones(Q.shape[0], device=device) / Q.shape[0]
            c = torch.ones(Q.shape[1], device=device) / (pl_module.world_size() * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            pl_module.reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                pl_module.reduce(curr_sum)

            prob = (Q / torch.sum(Q, dim=0, keepdim=True)).float()
            prob = prob.t()

            return prob


class HuberAuxiliaryLoss(object):
    def __init__(self, tau):
        self.tau = tau

    def run_pn(self, targets, preds, pseudo_label):
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

        mask_ignore = (pseudo_label == -1).float()
        mask = (1 - mask_ignore).view(b,n,1) * (1 - mask_ignore).view(b,1,n)

        return (diff_target * diff_pred * mask).sum(-1)

    def run_pp(self, targets, preds, pseudo_label):

        def _surrogate_fn_pp(x):
            return (x <= 0).float()

        def _surrogate_target(x):
            return (x <= 0).float()

        b, n = preds.shape
        diff_target = _surrogate_target(targets.view(b,n,1) - targets.view(b,1,n))
        diff_pred = _surrogate_fn_pp(preds.view(b,n,1) - preds.view(b,1,n))

        mask_ignore = (pseudo_label == -1).float()
        mask = (1 - mask_ignore).view(b,n,1) * (1 - mask_ignore).view(b,1,n)

        return (diff_target * diff_pred * mask).sum(-1)


def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
