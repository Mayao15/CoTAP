# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import torch

from torch import nn as nn
from torchvision.ops import roi_align


class InnerSampleLoss(nn.Module):

    def __init__(self, cfg):
        super(InnerSampleLoss, self).__init__()
        self.cfg = cfg

    def preprocess_feats(self, feats_gc, feats_lc=None):
        nmb_crops = self.cfg.nmb_crops
        bs = feats_gc.shape[0] // 2 // nmb_crops[0]
        shape_gc = feats_gc.shape[1:]

        feats_gc, feats_gc_pos = feats_gc.chunk(2)
        feats_gc = torch.cat([
            feats_gc.reshape(nmb_crops[0], bs, *shape_gc),
            feats_gc_pos.reshape(nmb_crops[0], bs, *shape_gc)
        ], dim=1)

        if feats_lc is None:
            return feats_gc

        shape_lc = feats_lc.shape[1:]
        feats_lc, feats_lc_pos = feats_lc.chunk(2)
        feats_lc = torch.cat([
            feats_lc.reshape(nmb_crops[1], bs, *shape_lc),
            feats_lc_pos.reshape(nmb_crops[1], bs, *shape_lc)
        ], dim=1)

        return feats_gc, feats_lc

    def forward_patch(self, outputs_gc_stu, outputs_gc_tea, outputs_lc_stu, \
        bbox_dict, masks=None, prefix='', head_idx=0, **kwargs):
        if not 'feat_%d_sk'%head_idx in outputs_gc_tea.keys():
            return [{
                'name': 'loss_%s_patch_inner'%prefix,
                'loss': 0 * outputs_gc_stu['feat_%d'%head_idx].mean(),
                'weight': self.cfg.weight_patch_inner
            }]

        gc_feat_stu = outputs_gc_stu['feat_%d'%head_idx]
        gc_feat_tea_sk = outputs_gc_tea['feat_%d_sk'%head_idx]
        lc_feat_stu = outputs_lc_stu['feat_%d'%head_idx]
        attn = outputs_gc_tea['attn']
        bboxes_gc = bbox_dict['gc']
        bboxes_all = bbox_dict['all']

        gc_feat_stu, lc_feat_stu = self.preprocess_feats(gc_feat_stu, lc_feat_stu)
        gc_feat_tea = self.preprocess_feats(gc_feat_tea_sk)

        loss_patch = 0 * gc_feat_stu.mean()
        n_gc, n_lc = gc_feat_stu.shape[0], lc_feat_stu.shape[0]
        bs, d, h_gc, w_gc = gc_feat_stu[0].shape
        h_lc, w_lc = lc_feat_stu[0].shape[-2:]
        kernel_size = self.cfg.roi_align_kernel_size
        patch_size = self.cfg.patch_size
        nmb_crops = self.cfg.nmb_crops

        gc_feat_tea = gc_feat_tea.reshape(n_gc, bs*h_gc*w_gc, d)
        gc_feat_stu = gc_feat_stu.permute(0,1,3,4,2).reshape(n_gc, bs*h_gc*w_gc, d)
        lc_feat_stu = lc_feat_stu.permute(0,1,3,4,2).reshape(n_lc, bs*h_lc*w_lc, d)
        
        # _keep = attn.reshape(n_gc, bs, 1, h_gc, w_gc)

        _keep = torch.ones(n_gc, bs, 1, h_gc, w_gc, device=gc_feat_stu.device)
        if masks is not None:
            _keep *= 1 - masks.view(n_gc, bs, 1, h_gc, w_gc).float()
        if attn is not None:
            _keep *= attn.view(n_gc, bs, 1, h_gc, w_gc).float()

        # swav loss computation
        for i, crop_id in enumerate(self.cfg.crops_for_assign):

            # 6. Roi align cluster assignments
            downsampled_crop_boxes = torch.unbind(bboxes_gc[:, crop_id] / patch_size)
            q_reshaped = gc_feat_tea[crop_id].reshape(bs, h_gc, w_gc, d).permute(0, 3, 1, 2)
            aligned_soft_clusters = roi_align(q_reshaped, downsampled_crop_boxes, kernel_size, aligned=True)  # (bs * num_crops, 7, 7, 2048)
            
            gc_keep = _keep[crop_id]  # select attn for crop_id
            aligned_keep = roi_align(gc_keep, downsampled_crop_boxes, kernel_size, aligned=True)
            thresholded_keep = (aligned_keep >= 1.0)  # Make mask 1-0

            # 7 .cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
                if v in self.cfg.crops_for_assign:
                    # Code prediction from other global crop
                    out = gc_feat_stu[v]
                    spatial_res = h_gc
                else:
                    # Code prediction from local crop
                    lc_index = v - nmb_crops[0]
                    out = lc_feat_stu[lc_index]
                    spatial_res = h_lc
                # Roi align cluster predictions
                aligned_out = roi_align(
                    out.reshape(bs, spatial_res, spatial_res, -1).permute(0, 3, 1, 2), 
                    torch.unbind(bboxes_all[:, v, crop_id].unsqueeze(1) / patch_size),
                    kernel_size,
                    aligned=True
                )

                aligned_log_p = torch.log_softmax(aligned_out / self.cfg.temperature_patch, dim=1)
                aligned_q = aligned_soft_clusters[v::np.sum(nmb_crops)]
                # Mask cross entropy if attn mask was passed
                keep_v = thresholded_keep[v::np.sum(nmb_crops)].squeeze().float()
                subloss -= torch.sum(torch.sum(aligned_q * aligned_log_p, dim=1) * keep_v) / torch.sum(keep_v).clamp(1)
                
            loss_patch += subloss / (np.sum(nmb_crops) - 1)

        loss_patch /= len(self.cfg.crops_for_assign)
        losses = [
            {
                'name': 'loss_%s_patch_inner'%prefix,
                'loss': loss_patch.mean(),
                'weight': self.cfg.weight_patch_inner
            }
        ]

        return losses

    def forward_cls(self, outputs_gc_stu, outputs_gc_tea, outputs_lc_stu, \
        prefix='', head_idx=0, **kwargs):

        if not 'cls_%d_sk'%head_idx in outputs_gc_tea.keys():
            return [{
                'name': 'loss_%s_cls_inner'%prefix,
                'loss': 0 * outputs_gc_stu['cls_%d'%head_idx].mean(),
                'weight': self.cfg.weight_cls_inner
            }]

        gc_feat_stu = outputs_gc_stu['cls_%d'%head_idx]
        gc_feat_tea_sk = outputs_gc_tea['cls_%d_sk'%head_idx]
        lc_feat_stu = outputs_lc_stu['cls_%d'%head_idx]

        gc_feat_stu, lc_feat_stu = self.preprocess_feats(gc_feat_stu, lc_feat_stu)
        gc_feat_tea = self.preprocess_feats(gc_feat_tea_sk)

        loss_cls = 0 * gc_feat_stu.mean()
        nmb_crops = self.cfg.nmb_crops

        # swav loss computation
        for i, crop_id in enumerate(self.cfg.crops_for_assign):
            q = gc_feat_tea[crop_id]
            subloss = 0
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
                if v in self.cfg.crops_for_assign:
                    # Code prediction from other global crop
                    out = gc_feat_stu[v]
                else:
                    # Code prediction from local crop
                    lc_index = v - nmb_crops[0]
                    out = lc_feat_stu[lc_index]
                log_p = torch.log_softmax(out / self.cfg.temperature_cls, dim=1)
                # Mask cross entropy if attn mask was passed
                subloss -= torch.mean(torch.sum(q * log_p, dim=1))

            loss_cls += subloss / (np.sum(nmb_crops) - 1)

        loss_cls /= len(self.cfg.crops_for_assign)
        losses = [
            {
                'name': 'loss_%s_cls_inner'%prefix,
                'loss': loss_cls.mean(),
                'weight': self.cfg.weight_cls_inner
            }
        ]

        return losses
    
    def forward(self, outputs_stu, outputs_tea, masks, lc_outputs_stu, \
        bboxes, pl_module, **kwargs):

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

        losses = []
        if self.cfg.weight_patch_inner > 0:
            losses += self.forward_patch(head_idx=self.cfg.head_idx_patch, **args)
        if self.cfg.weight_cls_inner > 0:
            losses += self.forward_cls(head_idx=self.cfg.head_idx_cls, **args)

        return losses


def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
