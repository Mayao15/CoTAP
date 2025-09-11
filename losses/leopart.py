# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
from turtle import forward
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import time

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import distributed as dist
from torch import nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torchvision.ops import roi_align
from typing import Callable, Optional, List, Any, Iterator, Tuple, Dict

# from experiments.utils import PredsmIoUKmeans, process_attentions, cosine_scheduler
# from src.vit import vit_small, vit_base, vit_large


class LeopartLoss(nn.Module):

    def __init__(self, cfg):
        super(LeopartLoss, self).__init__()
        self.cfg = cfg
        self.crops_for_assign = cfg.crops_for_assign
        self.patch_size = cfg.patch_size
        self.roi_align_kernel_size = cfg.roi_align_kernel_size
        self.nmb_crops = cfg.nmb_crops
        self.temperature = cfg.temperature
        self.sinkhorn_iterations = cfg.sinkhorn_iterations
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, 
    #     gc_teacher_output: torch.Tensor, 
    #     gc_student_output: torch.Tensor,
    #     lc_student_output: torch.Tensor, 
    #     gc_teacher_emb: torch.Tensor,
    #     bboxes: Dict,
    #     bs: int,
    #     gc_spatial_res: int, 
    #     attn_hard: torch.Tensor = None) -> float:

    def forward(self, feats, cluster, attn,
        feats_tea, cluster_tea, attn_tea, 
        queue_cluster, bbox_dict, pl_module=None, cluster_prob=None
    ):

        n_gc, n_lc = feats[0].shape[0], feats[1].shape[0]
        bs, d_feat, h, w = feats[0][0].shape
        d_cluster = cluster[0][0].shape[1]
        h_lc, w_lc = feats[1][0].shape[-2:]

        gc_teacher_output = cluster_tea[0].permute(0,1,3,4,2).reshape(n_gc*bs*h*w, d_cluster)
        gc_student_output = cluster[0].permute(0,1,3,4,2).reshape(n_gc*bs*h*w, d_cluster)
        lc_student_output = cluster[1].permute(0,1,3,4,2).reshape(n_lc*bs*h_lc*w_lc, d_cluster)
        attn_hard = attn_tea.reshape(n_gc*bs, 1, h, w)
        bboxes = bbox_dict

        # inputs = torch.load('debug_data.pth', map_location='cuda:0')
        # gc_teacher_output = inputs['gc_teacher_output']
        # gc_student_output = inputs['gc_student_output']
        # lc_student_output = inputs['lc_student_output']
        # bboxes = inputs['bboxes']
        # attn_hard = inputs['attn_hard']

        # print_tensor(gc_teacher_output, 'gc_teacher_output')
        # print_tensor(gc_student_output, 'gc_student_output')
        # print_tensor(lc_student_output, 'lc_student_output')
        # print_tensor(attn_hard, 'attn_hard')

        # 4. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):

            with torch.no_grad():
                # Select spatial cluster preds for global crop with crop_id
                out = gc_teacher_output[bs * h * w * crop_id:bs * h * w * (crop_id + 1)]
                num_spatial_vectors_for_pred = out.shape[0]
                if cluster_prob is not None:
                    q = cluster_prob[crop_id].permute(0,2,3,1).reshape(num_spatial_vectors_for_pred, -1)
                else:
                    if queue_cluster is not None:
                        out = torch.cat([queue_cluster, out], dim=0)
                    q = self.sinkhorn(out, self.sinkhorn_iterations, pl_module)[-num_spatial_vectors_for_pred:]

            # 6. Roi align cluster assignments
            q_reshaped = q.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
            downsampled_current_crop_boxes = torch.unbind(bboxes["gc"][:, crop_id] / self.patch_size)

            aligned_soft_clusters = roi_align(q_reshaped, downsampled_current_crop_boxes,
                                              self.roi_align_kernel_size, aligned=True)  # (bs * num_crops, 7, 7, 2048)

            if attn_hard is not None:
                # 6.5 Roi align mask
                gc_hard_mask = attn_hard[bs * crop_id: bs * (crop_id+1)]  # select attn for crop_id
                aligned_mask = roi_align(gc_hard_mask, downsampled_current_crop_boxes, \
                    self.roi_align_kernel_size, aligned=True)
                thresholded_mask = (aligned_mask >= 1.0)  # Make mask 1-0

            # 7 .cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                if v in self.crops_for_assign:
                    # Code prediction from other global crop
                    out = gc_student_output[bs * h * w * v:bs * h * w * (v + 1)]
                    spatial_res = h
                else:
                    # Code prediction from local crop
                    lc_index = v - self.nmb_crops[0]
                    out = lc_student_output[bs * h_lc * w_lc * lc_index:bs * h_lc * w_lc * (lc_index + 1)]
                    spatial_res = h_lc
                # Roi align cluster predictions
                aligned_out = roi_align(out.reshape(bs, spatial_res, spatial_res, -1).permute(0, 3, 1, 2),
                                        torch.unbind(bboxes["all"][:, v, crop_id].unsqueeze(1) / self.patch_size),
                                        self.roi_align_kernel_size,
                                        aligned=True)

                aligned_p = self.softmax(aligned_out / self.temperature)
                aligned_q = aligned_soft_clusters[v::np.sum(self.nmb_crops)]
                # Mask cross entropy if attn mask was passed
                if attn_hard is not None:
                    mask = thresholded_mask[v::np.sum(self.nmb_crops)].squeeze().float()
                    if torch.sum(mask).item()!=0:
                        subloss -= torch.sum(torch.sum(aligned_q * torch.log(aligned_p), dim=1) * mask) / torch.sum(mask)
                else:
                    # otherwise apply loss on all spatial tokens.
                    subloss -= torch.mean(torch.sum(aligned_q * torch.log(aligned_p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)

        loss /= len(self.crops_for_assign)

        losses = [{'name': 'loss_leopart', 'loss': loss, 'weight': 1}]

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

def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)

