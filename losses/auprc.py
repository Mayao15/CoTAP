# from math import perm
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from utils import *


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def norm(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-10)


def tensor_correlation(a, b):
    if len(a.shape) == 4:
        return torch.einsum("nchw,ncij->nhwij", norm(a), norm(b))
    # elif len(a.shape) == 3:
    #     return torch.einsum("nci,ncj->nij", a, b)
    elif len(a.shape) == 2:
        return torch.einsum("nc,mc->nm", norm(a), norm(b))
    else:
        assert False

def sample(t: torch.Tensor, coords: torch.Tensor):
    if len(coords.shape) == 4:
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
    elif len(coords.shape) == 3:
        return F.grid_sample(t, coords.unsqueeze(1), padding_mode='border', align_corners=True).squeeze(1)
    else:
        assert False

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size

def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class SLAPLoss(nn.Module):

    def __init__(self, cfg):
        super(SLAPLoss, self).__init__()
        self.cfg = cfg

        if self.cfg.weight_fn == 'linear':
            self.weight_fn = self.weight_fn_linear
        elif self.cfg.weight_fn == 'step':
            self.weight_fn = self.weight_fn_step
        elif self.cfg.weight_fn == 'hinge':
            self.weight_fn = self.weight_fn_hinge
        else:
            assert False

        temp = self.cfg.temp
        if cfg.sur_loss == 'hinge':
            self.aux = [HingeAuxiliaryLoss(t) for t in temp]
        elif cfg.sur_loss == 'huber':
            self.aux = [HuberAuxiliaryLoss(t) for t in temp]

    def slap_loss_cls(self, targets, preds, pseudo_label=None, idx=0, **kwargs):

        assert not targets.requires_grad and preds.requires_grad

        with torch.no_grad():
            weight = self.weight_fn(targets)
            mask_pos = (pseudo_label == 1).float()
            weight = (1 - mask_pos) * self.cfg.alpha[idx] * weight + mask_pos * weight

        with torch.no_grad():
            loss_pp = self.aux[idx].run_pp(targets, preds)

        loss_pn = self.aux[idx].run_pn(targets, preds)
        loss_pn *= self.cfg.mul_similarity

        loss = loss_pn / (1e-5 + loss_pp)
        g_loss = (loss / (1 + loss)) * weight
        g_loss = g_loss.sum(-1) / (1e-5 + weight.sum(-1))

        return g_loss.mean()

    def slap_loss_spatial(self, targets, preds, cls_targets, \
            pseudo_label=None, idx=1, **kwargs):

        assert not targets.requires_grad and preds.requires_grad

        with torch.no_grad():
            weight = self.weight_fn(targets)
            mask_pos = (pseudo_label == 1).float()
            weight = (1 - mask_pos) * self.cfg.alpha[idx] * weight + mask_pos * weight

        loss_pp = self.aux[idx].run_pp_acc(targets, preds)
        loss_pn = self.aux[idx].run_pn_acc(targets, preds)

        # loss_pn *= self.cfg.mul_similarity

        loss = loss_pn / (1e-5 + loss_pp)
        g_loss = (loss / (1 + loss)) * weight
        g_loss = g_loss.sum(-1) / (1e-5 + weight.sum(-1))

        return g_loss.mean()

    def get_img_emb(self, x, attn):
        emb = F.adaptive_avg_pool2d(x * attn, (1,1)).reshape(x.shape[0], -1)
        return emb

    def get_image_sim(self, feats, feats_tea, attn, pl_module):

        emb_query, emb_cad = self.get_img_emb(feats, attn.detach()).chunk(2)
        emb_query_tea, emb_cad_tea = self.get_img_emb(feats_tea, attn.detach()).chunk(2)
        n, d = emb_query.shape[:2]
        global_rank = pl_module.global_rank

        def gather(x):
            x_gt = pl_module.all_gather(x)
            return torch.cat([
                x_gt[:global_rank], x.unsqueeze(0), x_gt[global_rank+1:]
            ], dim=0)

        emb_query = gather(emb_query).view(-1, d)
        emb_cad = gather(emb_cad).view(-1, d)
        emb_query_tea = gather(emb_query_tea).view(-1, d)
        emb_cad_tea = gather(emb_cad_tea).view(-1, d)

        sim = tensor_correlation(emb_query, emb_cad)
        sim_tea = tensor_correlation(emb_query_tea, emb_cad_tea)

        pseudo_label = torch.diag(torch.ones(len(emb_query), device=sim_tea.device))
        # pseudo_label = torch.ones((n, n), device=sim_tea.device)

        return sim, sim_tea, pseudo_label    

    def forward_cross_image(self, outputs_stu, outputs_tea, pl_module, alpha=0.1, head_idx=1, **kwargs):
        attn = outputs_tea['attn']
        spatial_feat_stu = outputs_stu['feat_%d'%head_idx]
        spatial_feat_tea = outputs_tea['feat_%d'%head_idx]
        # cls_feat_stu = outputs_stu['cls_%d'%head_idx]
        # cls_feat_tea = outputs_tea['cls_%d'%head_idx]

        img_sim, img_sim_tea, img_pseudo_label = \
            self.get_image_sim(spatial_feat_stu, spatial_feat_tea, attn, pl_module)
            # self.get_image_sim(cls_feat_stu, cls_feat_tea, attn, pl_module)
        loss_image = self.slap_loss_cls(img_sim_tea, img_sim, \
            pseudo_label=img_pseudo_label, idx=0, alpha=alpha)

        losses = [
            {
                'name': 'loss_%s_image_cross'%self.cfg.prefix,
                'loss': loss_image.mean(),
                'weight': self.cfg.weight_image_cross
            }
        ]

        return losses

    def get_patch_sim(self, feats_stu, feats_tea, cls_feats_stu, cls_feats_tea, attn, keep):
        bs, d, h, w = feats_stu.shape
        bs = bs // 2

        feats_query_stu, feats_cad_stu = feats_stu.chunk(2)
        feats_query_tea, feats_cad_tea = feats_tea.chunk(2)
        cls_query_stu, cls_cad_stu = cls_feats_stu.chunk(2)
        cls_query_tea, cls_cad_tea = cls_feats_tea.chunk(2)
        attn_query, attn_cad = attn.chunk(2)
        keep_query, keep_cad = keep.chunk(2)
        idx_keep = (attn_query.view(-1) > 0) & (keep_query.view(-1) > 0)

        def get_sim(f1, f2, c1, c2, perm=None):
            if perm is not None:
                _f2, _c2 = f2[perm], c2[perm]
            else:
                _f2, _c2 = f2, c2
            spatial_sim = tensor_correlation(f1, _f2).view(bs*h*w, h*w)
            cls_sim = (norm(c1) * norm(_c2)).sum(-1)
            cls_sim = cls_sim.view(bs, 1, 1).repeat(1, h*w, h*w).reshape(-1, h*w)
            return spatial_sim[idx_keep], cls_sim[idx_keep]

        # reduce weights of pairs from diff images
        # pseudo label:  1 for pos, 0 for neg, -1 for ignore
        def get_pseudo_label(attn1, attn2, keep1, keep2, perm=None):
            if perm is not None:
                _attn2, _keep2 = attn2[perm], keep2[perm]
            else:
                _attn2, _keep2 = attn2, keep2
            _keep1 = keep1.view(bs,h*w) * attn1.view(bs,h*w)
            _keep2 = _keep2.view(bs,h*w) * _attn2.view(bs,h*w)
            pseudo_label = torch.einsum("nx,ny->nxy", _keep1, _keep2)
            pseudo_label = pseudo_label.long().reshape(bs*h*w, h*w)
            pseudo_label = 2 * pseudo_label - 1 if perm is None else pseudo_label - 1
            return pseudo_label[idx_keep]

        spatial_sim_stu, cls_sim_stu = [], []
        spatial_sim_tea, cls_sim_tea = [], []
        pseudo_label = []
        for perm in [None] + [super_perm(bs, feats_stu.device) for _ in range(self.cfg.n_rand)]:
            spatial_sim_stu_, cls_sim_stu_ = \
                get_sim(feats_query_stu, feats_cad_stu, cls_query_stu, cls_cad_stu, perm)
            spatial_sim_tea_, cls_sim_tea_ = \
                get_sim(feats_query_tea, feats_cad_tea, cls_query_tea, cls_cad_tea, perm)
            pseudo_label_ = get_pseudo_label(attn_query, attn_cad, keep_query, keep_cad, perm)
            spatial_sim_stu.append(spatial_sim_stu_)
            spatial_sim_tea.append(spatial_sim_tea_)
            cls_sim_stu.append(cls_sim_stu_)
            cls_sim_tea.append(cls_sim_tea_)
            pseudo_label.append(pseudo_label_)

        spatial_sim_stu = torch.cat(spatial_sim_stu, dim=1)
        spatial_sim_tea = torch.cat(spatial_sim_tea, dim=1)
        cls_sim_stu = torch.cat(cls_sim_stu, dim=1)
        cls_sim_tea = torch.cat(cls_sim_tea, dim=1)
        pseudo_label = torch.cat(pseudo_label, dim=1)

        return spatial_sim_stu, spatial_sim_tea, cls_sim_stu, cls_sim_tea, pseudo_label

    def forward_cross_patch(self, outputs_stu, outputs_tea, masks, alpha=0.1, head_idx=1, **kwargs):
        attn = outputs_tea['attn']
        feat_stu = outputs_stu['feat_%d'%head_idx]
        feat_tea = outputs_tea['feat_%d'%head_idx] # bs * d * h * w
        cls_feat_stu = outputs_stu['cls_%d'%head_idx]
        cls_feat_tea = outputs_tea['cls_%d'%head_idx]

        spatial_sim_stu, spatial_sim_tea, cls_sim_stu, cls_sim_tea, pseudo_label = \
            self.get_patch_sim(feat_stu, feat_tea, cls_feat_stu, cls_feat_tea, attn, ~masks)
        loss_patch = self.slap_loss_spatial(spatial_sim_tea, spatial_sim_stu, \
            cls_sim_tea, pseudo_label, idx=1, alpha=alpha)

        losses = [
            {
                'name': 'loss_patch_cross',
                'loss': loss_patch.mean(),
                'weight': self.cfg.weight_patch_cross
            }
        ]

        return losses

    def forward(self, outputs_stu, outputs_tea, masks, pl_module):
        args = {
            'outputs_stu': outputs_stu,
            'outputs_tea': outputs_tea,
            'masks': masks,
            'pl_module': pl_module,
            'eps': 1e-5
        }
        losses = self.forward_cross_image(**args, head_idx=self.cfg.head_idx, alpha=0.1)
        losses += self.forward_cross_patch(**args, head_idx=self.cfg.head_idx)
        # losses += self.forward_cross_object(**args, head_idx=1, alpha=0.5)

        return losses

    def weight_fn_linear(self, x):
        return x

    def weight_fn_step(self, x):
        thres = self.cfg.weight_param
        return (x >= thres).float()

    def weight_fn_hinge(self, x):
        thres = self.cfg.weight_param
        return torch.clamp(x - thres, 0)


class HingeAuxiliaryLoss(object):
    def __init__(self, temp):
        self.temp = temp

    def run_pn(self, targets, preds):
        temp = self.temp

        def _surrogate_fn_pn(x):
            return torch.clamp(temp - x, 0)

        def _surrogate_fn_pn_acc(x_pos, x_neg, y_pos, y_neg):

            def get_suf_sum(x_tensor):
                pre_sum = torch.cumsum(x_tensor, 1)
                pre_sum = torch.cat([torch.zeros(len(pre_sum), 1).cuda(), pre_sum], 1)
                suf_sum = pre_sum[:, -1:] - pre_sum
                return suf_sum

            sort_idx = torch.sort(x_neg, 1)[1]
            x_neg = torch.gather(x_neg, 1, sort_idx)
            y_neg = torch.gather(y_neg, 1, sort_idx)
            idx = torch.searchsorted(x_neg, x_pos - temp)

            suf_sum_x_neg = get_suf_sum(x_neg)
            suf_sum_y_neg = get_suf_sum(y_neg)
            suf_sum_xy_neg = get_suf_sum(x_neg * y_neg)

            loss = 0
            loss += y_pos * torch.gather(suf_sum_x_neg, 1, idx)
            loss += (x_pos - temp) * torch.gather(suf_sum_y_neg, 1, idx)
            loss -= torch.gather(suf_sum_xy_neg, 1, idx)
            loss += y_pos * (temp - x_pos) * (y_neg.shape[1] - idx)

            # sort_idx = torch.sort(x_neg, 1)[1]
            # x_neg = torch.gather(x_neg, 1, sort_idx)
            # idx = torch.searchsorted(x_neg, x_pos - temp)

            # suf_sum_x_neg = get_suf_sum(x_neg)

            # loss = 0
            # loss += torch.gather(suf_sum_x_neg, 1, idx)
            # loss += (temp - x_pos) * (x_pos.shape[1] - idx)

            return loss

        def _solve_pn_acc(targets, preds, ans, is_first=False):
            if targets.shape[1] <= 100:
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

            _solve_pn_acc(targets, preds, ans, is_first=False)
            ans = torch.gather(ans, 1, torch.sort(idx, dim=1)[1])

            return ans / temp
            # return ans

        return solve_pn_acc(targets, preds)

    def run_pp(self, targets, preds):
        temp = self.temp

        def _surrogate_fn_pp(x):
            return (x <= 0).float()

        def _surrogate_fn_pp_acc(x, y):
            y = torch.sort(y, 1)[0]
            idx = torch.searchsorted(y, x)
            loss = (y.shape[1] - idx)
            return loss

        def _solve_pp_acc(targets, preds, ans):
            if targets.shape[1] <= 100:
                n = targets.shape[0]
                loss = _surrogate_fn_pp(preds.view(n, -1, 1) - preds.view(n, 1, -1))
                ans += (loss * (targets.view(n, -1, 1) <= targets.view(n, 1, -1)).float()).sum(-1)
                return

            mid = preds.shape[1] // 2
            ans[:, :mid] += _surrogate_fn_pp_acc(preds[:, :mid], preds[:, mid:])
            _solve_pp_acc(targets[:, :mid], preds[:, :mid], ans[:, :mid])
            _solve_pp_acc(targets[:, mid:], preds[:, mid:], ans[:, mid:])

        def solve_pp_acc(targets, preds):
            ans = torch.zeros_like(preds).cuda()
            idx = torch.sort(targets, dim=1)[1]
            targets = torch.gather(targets, 1, idx)
            preds = torch.gather(preds, 1, idx)
            _solve_pp_acc(targets, preds, ans)

            ans = torch.gather(ans, 1, torch.sort(idx, dim=1)[1])
            return ans

        return solve_pp_acc(targets, preds)


class HuberAuxiliaryLoss(object):
    def __init__(self, temp):
        self.temp = temp

    def run_pn(self, targets, preds):

        def _surrogate_fn_pn(x):
            x = torch.clamp(x / self.temp, max=1)
            return torch.where(
                x >= 0,
                (x - 1)*(x - 1),
                (-2 * x + 1)
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


    def run_pn_acc(self, targets, preds):
        tau = self.temp

        def _surrogate_fn_pn(x):
            x = torch.clamp(x / tau, max=1)
            return torch.where(
                x >= 0,
                (x - 1)*(x - 1),
                (-2 * x + 1)
            )

        def _surrogate_fn_pn_acc(x_pos, x_neg, y_pos, y_neg):

            def get_pre_suf_sum(x_tensor, idx_pre, idx_suf):
                pre_sum = torch.cat([torch.zeros(len(x_tensor), 1, \
                    device=x_tensor.device), torch.cumsum(x_tensor, 1)], 1)
                suf_sum = pre_sum[:, -1:] - pre_sum
                suf_selected = torch.gather(suf_sum, 1, idx_suf)
                pre_selected = torch.gather(pre_sum, 1, idx_pre)\
                     - torch.gather(pre_sum, 1, idx_suf)
                return pre_selected, suf_selected

            sort_idx = torch.sort(x_neg, 1)[1]
            x_neg = torch.gather(x_neg, 1, sort_idx)
            y_neg = torch.gather(y_neg, 1, sort_idx)
            k = torch.searchsorted(x_neg, x_pos - tau, right=True)
            t = torch.searchsorted(x_neg, x_pos, right=False)

            n = x_neg.shape[1]
            pre_x, suf_x = get_pre_suf_sum(x_neg, t, k)
            pre_y, suf_y = get_pre_suf_sum(y_neg, t, k)
            pre_xy, suf_xy = get_pre_suf_sum(x_neg * y_neg, t, k)
            pre_x_sq, _ = get_pre_suf_sum(x_neg**2, t, k)
            pre_x_sq_y, _ = get_pre_suf_sum(x_neg**2 * y_neg, t, k)

            # print_tensor(x_pos, 'x_pos')
            # print_tensor(x_neg, 'x_neg')
            # print_tensor(y_pos, 'y_pos')
            # print_tensor(y_neg, 'y_neg')
            # print_tensor(pre_x, 'pre_x')
            # print_tensor(suf_x, 'suf_x')
            # print_tensor(pre_y, 'pre_y')
            # print_tensor(suf_y, 'suf_y')
            # print_tensor(pre_xy, 'pre_xy')
            # print_tensor(suf_xy, 'suf_xy')
            # print_tensor(pre_x_sq, 'pre_x_sq')
            # print_tensor(pre_x_sq_y, 'pre_x_sq_y')

            loss1 = (-2 / tau * x_pos + 1) * ((n - k) * y_pos - suf_y)
            loss1 += 2 / tau * (y_pos * suf_x - suf_xy)

            # loss_debug = (-2 * (x_pos[0][0] - x_neg[0]) / tau + 1) * (y_pos[0][0] - y_neg[0])
            # # loss_debug = (-2 * (x_pos[0][0] - x_neg[0]) / tau + 1) * (y_pos[0][0]) \
            # #     + (-2 * (x_pos[0][0]) / tau + 1) * ( - y_neg[0])
            # loss_debug = loss_debug[k[0][0]:].sum()
            # print(k[0][0])
            # print_tensor(loss_debug, 'loss_debug')
            # print_tensor(loss1[0][0], 'loss1')
            # assert False

            loss2 = x_pos**2 * ((t - k) * y_pos - pre_y)
            loss2 += y_pos * pre_x_sq - pre_x_sq_y
            loss2 += -2 * x_pos * y_pos * pre_x + 2 * x_pos * pre_xy

            loss = loss1 + loss2 / tau**2

            # loss += y_pos * torch.gather(suf_sum_x_neg, 1, idx)
            # loss += (x_pos - temp) * torch.gather(suf_sum_y_neg, 1, idx)
            # loss -= torch.gather(suf_sum_xy_neg, 1, idx)
            # loss += y_pos * (temp - x_pos) * (y_neg.shape[1] - idx)

            # sort_idx = torch.sort(x_neg, 1)[1]
            # x_neg = torch.gather(x_neg, 1, sort_idx)
            # idx = torch.searchsorted(x_neg, x_pos - temp)

            # suf_sum_x_neg = get_suf_sum(x_neg)

            # loss = 0
            # loss += torch.gather(suf_sum_x_neg, 1, idx)
            # loss += (temp - x_pos) * (x_pos.shape[1] - idx)

            return loss

        def _solve_pn_acc(targets, preds, ans):
            if targets.shape[1] <= 100:
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
            targets = targets.float()
            preds = preds.float()
            ans = 0 * preds
            # _preds = preds / self.temp
            idx = torch.sort(-targets, dim=1)[1]
            targets = torch.gather(targets, 1, idx)
            preds = torch.gather(preds, 1, idx)

            _solve_pn_acc(targets, preds, ans)
            ans = torch.gather(ans, 1, torch.sort(idx, dim=1)[1])
            # ans = torch.clamp(ans, 0)

            return ans
            # return ans

        return solve_pn_acc(targets, preds)

    @torch.no_grad()
    def run_pp_acc(self, targets, preds):
        def _surrogate_fn_pp(x):
            return (x <= 0).float()

        def _surrogate_fn_pp_acc(x, y):
            y = torch.sort(y, 1)[0]
            idx = torch.searchsorted(y, x)
            loss = (y.shape[1] - idx)
            return loss

        def _solve_pp_acc(targets, preds, ans):
            if targets.shape[1] <= 100:
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

    @torch.no_grad()
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


class BinaryCrossEntropyLoss(SLAPLoss):

    def __init__(self, cfg):
        super(BinaryCrossEntropyLoss, self).__init__(cfg)
        
    def helper(self, targets, preds, **kwargs):

        assert not targets.requires_grad and preds.requires_grad
        
        targets = 0.5 * (targets + 1)
        preds = 0.5 * (preds + 1)

        ce_loss = - targets * torch.log(1e-6 + preds) - (1 - targets) * torch.log(1e-6 + (1 - preds))
        return ce_loss.mean()


class DiscreteAUPRCLoss(SLAPLoss):

    def __init__(self, cfg):
        super(DiscreteAUPRCLoss, self).__init__(cfg)

    def helper(self, targets, preds, **kwargs):
        targets = (targets > 0).long()
        return super().helper(targets, preds, **kwargs)


def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
