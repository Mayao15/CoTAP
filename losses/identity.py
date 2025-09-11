# from math import perm
import torch
import torch.nn.functional as F

from utils import *


class _IdentitySingleLoss(nn.Module):

    def __init__(self, cfg):
        super(_IdentitySingleLoss, self).__init__()
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.topk = 10

    def forward_self(self, outputs_stu, outputs_tea, masks=None, pl_module=None, prefix='', head_idx=0, **kwargs):
        if not 'feat_%d'%head_idx in outputs_stu.keys():
            return []

        if not 'feat_%d_sk'%head_idx in outputs_tea.keys():
            return [{
                'name': 'loss_%s_patch_rebuild'%prefix,
                'loss': 0 * outputs_stu['feat_%d'%head_idx].mean(),
                'weight': self.cfg.weight_patch_rebuild
            }]

        patch_feat_stu = outputs_stu['feat_%d'%head_idx]
        patch_feat_tea_sk = outputs_tea['feat_%d_sk'%head_idx]
        # attn = outputs_tea['attn']

        loss_patch = 0 * patch_feat_stu.mean()
        loss_rebuild = 0 * patch_feat_stu.mean()
        bs, d, h, w = patch_feat_stu.shape

        patch_target_softmax = patch_feat_tea_sk.view(bs, h, w, d).permute(0,3,1,2)
        patch_pred_log_softmax = torch.log_softmax(patch_feat_stu / self.cfg.temperature, dim=1)
        if masks is not None:
            _masks = masks.view(bs, 1, h, w).float()
            _keep = 1 - _masks
            # if attn is not None:
            #     _keep *= attn
            #     _masks *= attn
            if torch.sum(_keep).item() != 0:
                loss_patch -= torch.sum(torch.sum(_keep * patch_target_softmax \
                    * patch_pred_log_softmax, dim=1)) / torch.sum(_keep).clamp(1)

            if torch.sum(_masks).item() != 0:
                loss_rebuild -= torch.sum(torch.sum(_masks * patch_target_softmax \
                    * patch_pred_log_softmax, dim=1)) / torch.sum(_masks).clamp(1)
        else:
            loss_patch -= torch.mean(torch.sum(patch_target_softmax \
                * patch_pred_log_softmax, dim=1))

        losses = [
            {
                'name': 'loss_%s_patch_self'%prefix,
                'loss': loss_patch.mean(),
                'weight': self.cfg.weight_patch_self
            },
            {
                'name': 'loss_%s_patch_rebuild'%prefix,
                'loss': loss_rebuild.mean(),
                'weight': self.cfg.weight_patch_rebuild
            }
        ]

        return losses

    def forward(self, **kwargs):
        return self.forward_self(**kwargs)


class IdentityLoss(nn.Module):

    def __init__(self, cfg):
        super(IdentityLoss, self).__init__()
        self.cfg = cfg
        self.loss_fn_global = _IdentitySingleLoss(cfg)

    def forward(self, outputs_stu, outputs_tea, masks, pl_module, **kwargs):

        losses = []  
        args = {
            'outputs_stu': outputs_stu,
            'outputs_tea': outputs_tea,
            'head_idx': self.cfg.head_idx_patch,
            'masks': masks,
            'pl_module': pl_module,
            'prefix': self.cfg.prefix + '_global',
            'eps': 1e-5
        }
        losses += self.loss_fn_global(**args)

        # if lc_outputs_tea is not None:
        #     args_lc = {
        #         'outputs_stu': lc_outputs_stu,
        #         'outputs_tea': lc_outputs_tea,
        #         'pl_module': pl_module,
        #         'prefix': self.cfg.prefix + '_local',
        #         'eps': 1e-5
        #     }
        #     losses += self.loss_fn_local(**args_lc)

        return losses
