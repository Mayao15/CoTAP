import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from .identity import IdentityLoss
from .inner_sample import InnerSampleLoss
from .intra_sample import IntraSampleLoss
from .kernel_align import KernelAlignLoss, EntropyMaxLoss
from .spatial_regularization import SpatialRegularizationLoss


class LossWrapper(nn.Module):
    def __init__(self, cfg):
        super(LossWrapper, self).__init__()
        self.cfg = cfg
        self._init_shared_module(cfg.share)

        loss_dict = {
            'Identity': IdentityLoss,
            'InnerSample': InnerSampleLoss,
            'IntraSample': IntraSampleLoss,
            'KernelAlign': KernelAlignLoss,
            'EntropyMax': EntropyMaxLoss,
            'SpatialRegularization': SpatialRegularizationLoss,
        }
        self.loss_fn = []
        for k in cfg.keys():
            if k == 'share':
                continue
            cfg[k].update(cfg.share)
            self.loss_fn.append(loss_dict[cfg[k].name](cfg[k]))
        self.loss_fn = nn.ModuleList(self.loss_fn)

    def _init_shared_module(self, cfg):
        self.patch_cluster_queue = MemoryBankPatch(cfg)
        self.cls_cluster_queue = MemoryBankCls(cfg)

    def _preprocess_patch(self, outputs_tea, lc_outputs_tea, pl_module):
        if not self.patch_cluster_queue.ready():
            return outputs_tea, lc_outputs_tea
        
        if not 'feat_%d'%self.cfg.share.head_idx_patch in outputs_tea.keys():
            return outputs_tea, lc_outputs_tea

        patch_feat_tea = outputs_tea['feat_%d'%self.cfg.share.head_idx_patch]
        temperature = self.cfg.share.sinkhonrn_temperature_patch
        bs, d, h, w = patch_feat_tea.shape

        patch_queue = self.patch_cluster_queue.queue_cluster
        patch_feat_tea = patch_feat_tea.permute(0,2,3,1).reshape(bs*h*w, d)
        num_patch = len(patch_feat_tea)
        if lc_outputs_tea is not None:
            lc_patch_feat_tea = lc_outputs_tea['feat_%d'%self.cfg.share.head_idx_patch]
            lc_patch_feat_tea = lc_patch_feat_tea.permute(0,2,3,1).reshape(-1, d)
            lc_num_patch = len(lc_patch_feat_tea)
            input_sinkhorn = torch.cat([patch_queue, patch_feat_tea, lc_patch_feat_tea], dim=0)
            out_sk = self.sinkhorn(input_sinkhorn, temperature, pl_module=pl_module)
            patch_feat_tea_sk = out_sk[-num_patch-lc_num_patch:-lc_num_patch]
            lc_patch_feat_tea_sk = out_sk[-lc_num_patch:]
            lc_outputs_tea['feat_%d_sk'%self.cfg.share.head_idx_patch] = lc_patch_feat_tea_sk
        else:
            input_sinkhorn = torch.cat([patch_queue, patch_feat_tea], dim=0)
            out_sk = self.sinkhorn(input_sinkhorn, temperature, pl_module=pl_module)
            patch_feat_tea_sk = out_sk[-num_patch:]
        outputs_tea['feat_%d_sk'%self.cfg.share.head_idx_patch] = patch_feat_tea_sk

        return outputs_tea

    def _postprocess_patch(self, outputs_tea, pl_module):
        if not 'feat_%d'%self.cfg.share.head_idx_patch in outputs_tea.keys():
            return
        patch_feat_tea = outputs_tea['feat_%d'%self.cfg.share.head_idx_patch]
        attn = outputs_tea['attn']
        self.patch_cluster_queue.update(patch_feat_tea, attn, pl_module)

    def _preprocess_cls(self, outputs_tea, lc_outputs_tea, pl_module):
        if not self.cls_cluster_queue.ready():
            return outputs_tea

        cls_feat_tea = outputs_tea['cls_%d'%self.cfg.share.head_idx_cls]            
        temperature = self.cfg.share.sinkhonrn_temperature_cls
        bs, d = cls_feat_tea.shape

        cls_queue = self.cls_cluster_queue.queue_cluster
        if lc_outputs_tea is not None:
            lc_cls_feat_tea = lc_outputs_tea['cls_%d'%self.cfg.share.head_idx_cls]
            bs_lc = len(lc_cls_feat_tea)
            input_sinkhorn = torch.cat([cls_queue, cls_feat_tea, lc_cls_feat_tea], dim=0)
            out_sk = self.sinkhorn(input_sinkhorn, temperature, pl_module=pl_module)
            cls_feat_tea_sk = out_sk[-(bs+bs_lc):-bs_lc]
            lc_cls_feat_tea_sk = out_sk[-bs_lc:]
            outputs_tea['cls_%d_sk'%self.cfg.share.head_idx_cls] = cls_feat_tea_sk
            lc_outputs_tea['cls_%d_sk'%self.cfg.share.head_idx_cls] = lc_cls_feat_tea_sk
        else:
            input_sinkhorn = torch.cat([cls_queue, cls_feat_tea], dim=0)
            cls_feat_tea_sk = self.sinkhorn(input_sinkhorn, temperature, pl_module=pl_module)[-bs:]
            outputs_tea['cls_%d_sk'%self.cfg.share.head_idx_cls] = cls_feat_tea_sk
        return outputs_tea

    def _postprocess_cls(self, outputs_tea, pl_module):
        cls_feat_tea = outputs_tea['cls_%d'%self.cfg.share.head_idx_cls]
        self.cls_cluster_queue.update(cls_feat_tea, pl_module)

    def forward(self, 
                outputs_stu, 
                outputs_tea, 
                lc_outputs_stu, 
                lc_outputs_tea, 
                masks, 
                num_subsets, 
                bboxes,
                kernel,
                samples,
                pl_module,
                **kwargs
        ):

        applied_subset = self.cfg.share.applied_subset_patch
        if applied_subset >= 0:
            if masks is not None:
                masks = masks.chunk(num_subsets)[applied_subset]
            # if outputs_tea['attn'] is not None:
            #     outputs_tea['attn'] = outputs_tea['attn'].chunk(num_subsets)[applied_subset]
            bboxes['gc'] = bboxes['gc'].chunk(num_subsets)[applied_subset]
            bboxes['all'] = bboxes['all'].chunk(num_subsets)[applied_subset]

        self._preprocess_patch(outputs_tea, lc_outputs_tea, pl_module)
        self._preprocess_cls(outputs_tea, lc_outputs_tea, pl_module)

        # for k in outputs_tea.keys():
        #     if isinstance(outputs_tea[k], torch.Tensor):
        #         print(k, outputs_tea[k].shape)
        # print('masks', masks.shape)
        # print('='*30)

        losses = []
        for loss_fn in self.loss_fn:
            losses += loss_fn(
                outputs_stu=outputs_stu,
                outputs_tea=outputs_tea,
                lc_outputs_stu=lc_outputs_stu,
                lc_outputs_tea=lc_outputs_tea,
                masks=masks,
                bboxes=bboxes,
                kernel=kernel,
                samples=samples,
                pl_module=pl_module,
                **kwargs
            )

        total_loss = 0
        for i in losses:
            if 'loss' in i.keys():
                assert torch.isnan(i['loss']).long().sum() == 0
                total_loss += i['weight'] * i['loss']

        self._postprocess_patch(outputs_tea, pl_module)
        self._postprocess_cls(outputs_tea, pl_module)
        
        return total_loss, losses

    @torch.no_grad()
    def sinkhorn(self, init_q, temperature, pl_module=None) -> torch.Tensor:
        device=init_q.device
        nmb_iters = self.cfg.share.sinkhorn_iterations
        with torch.no_grad():
            Q = torch.exp(init_q.detach() / temperature).t()
            
            # from .intra_sample import print_tensor
            # print_tensor(Q, 'Q')

            sum_Q = torch.sum(Q)
            if pl_module is not None:
                sum_Q = pl_module.reduce(sum_Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0], device=device)
            r = torch.ones(Q.shape[0], device=device) / Q.shape[0]
            world_size = pl_module.world_size() if pl_module is not None else 1
            c = torch.ones(Q.shape[1], device=device) / (world_size * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            if pl_module is not None:
                curr_sum = pl_module.reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                if pl_module is not None:
                    curr_sum = pl_module.reduce(curr_sum)

            prob = (Q / torch.sum(Q, dim=0, keepdim=True)).float()
            prob = prob.t()

        return prob

    def distributed_sinkhorn(self, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q
            device = Q.device
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            u = torch.zeros(Q.shape[0], device=device)
            r = torch.ones(Q.shape[0], device=device) / Q.shape[0]
            c = torch.ones(Q.shape[1], device=device) / (world_size * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
            
class MemoryBankPatch(nn.Module):
    def __init__(self, cfg, kernel_size=1):
        super(MemoryBankPatch, self).__init__()
        self._queue_cluster = None
        # self._kernel = None
        # self._kernel_size = kernel_size
        self._len_queue = cfg.len_patch_queue // cfg.world_size

    @torch.no_grad()
    def update(self, spatial_feat, attn, pl_module):
        spatial_feat = spatial_feat.data
        bs, d, h, w = spatial_feat.shape
        # kernel_size = self._kernel_size

        # kernel = norm(F.adaptive_avg_pool2d(\
        #     spatial_feat * attn, (kernel_size, kernel_size))) / kernel_size**2
        # n_keep_kernel = min(10, len(kernel))
        # keep_idx_kernel = torch.randperm(len(kernel))[:n_keep_kernel]

        if attn is not None:
            attn = attn.data
            masked_spatial_feat = torch.index_select(
                spatial_feat.permute(0,2,3,1).reshape(-1, d),
                dim=0,
                index=attn.reshape(-1).nonzero().view(-1)
            )
        else:
            masked_spatial_feat = spatial_feat.permute(0,2,3,1).reshape(-1, d)
        # masked_spatial_feat = spatial_feat.permute(0,2,3,1).reshape(-1, d)
        n_keep_queue = min(2048, len(masked_spatial_feat))
        keep_idx_queue = torch.randperm(len(masked_spatial_feat))[:n_keep_queue]

        if self._queue_cluster is None:
            # self._kernel = kernel[keep_idx_kernel] # bs * d * 7 * 7
            self._queue_cluster = masked_spatial_feat[keep_idx_queue]
        else:
            # self._kernel = torch.cat([
            #     kernel[keep_idx_kernel],
            #     self._kernel[:self._len_kernel - len(keep_idx_kernel)]
            # ])
            self._queue_cluster = torch.cat([
                masked_spatial_feat[keep_idx_queue],
                self._queue_cluster[:self._len_queue - len(keep_idx_queue)]
            ])
        
        # self._kernel = pl_module.broadcast(self._kernel, 0)
        # self._queue_cluster = pl_module.broadcast(self._queue_cluster, 0)
        # print('Patch:', self._queue_cluster.shape)

    def get_feat(self, spatial_feat):
        return F.conv2d(spatial_feat, self.kernel, padding=self._kernel_size//2)

    def ready(self):
        return self._queue_cluster is not None

    @property
    def kernel(self):
        return self._kernel

    @property
    def queue_cluster(self):
        return self._queue_cluster


class MemoryBankCls(nn.Module):
    def __init__(self, cfg):
        super(MemoryBankCls, self).__init__()
        self._queue_cluster = None
        self._len_queue = cfg.len_cls_queue // cfg.world_size

    @torch.no_grad()
    def update(self, cls_feat, pl_module):
        cls_feat = cls_feat.data
        bs, d = cls_feat.shape
        # cls_feat = pl_module.all_gather(cls_feat).view(-1, d)
        n_keep_queue = min(100, bs)
        keep_idx_queue = torch.randperm(len(cls_feat))[:n_keep_queue]
        
        if self._queue_cluster is None:
            self._queue_cluster = cls_feat[:self._len_queue]
        else:
            self._queue_cluster = torch.cat([
                cls_feat[keep_idx_queue],
                self._queue_cluster[:self._len_queue - len(cls_feat[keep_idx_queue])]
            ])
        # self._queue_cluster = pl_module.broadcast(self._queue_cluster, 0)
        # print('Cls:', self._queue_cluster.shape)

    @property
    def queue_cluster(self):
        return self._queue_cluster    

    def ready(self):
        return self._queue_cluster is not None
