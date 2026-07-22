# Adapted from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.nn.utils import weight_norm
from functools import partial
import sys
sys.path.append('../../../')

from datasets.transforms import normalize
from models.feature_regularization import AnisotropicDiffusion


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def forward_kernel_patch(self, x_in, kernel):
        N_heads = self.num_heads
        B, N, C = x_in.shape
        H = W = int(math.sqrt(N))
        M, ks = kernel.shape[:2]

        qkv = self.qkv(x_in).reshape(B, N, 3, N_heads, C // N_heads).permute(2, 0, 3, 1, 4)
        qkv_ker = self.qkv(kernel).reshape(M, ks, ks, 3, N_heads, C // N_heads).permute(3, 4, 0, 1, 2, 5)
        q, k_ker, v_ker = qkv[0], qkv_ker[1], qkv_ker[2] # N_heads * M * ks * ks * D

        q = q.reshape(B,N_heads,H,W,-1).permute(1,0,4,2,3)
        k_ker = k_ker.permute(0,1,4,2,3)
        attn_ker = torch.stack([F.conv2d(q[i], k_ker[i], \
            padding=ks//2) for i in range(N_heads)]) # N_heads * B * M * H * W
        attn_ker = attn_ker * (M ** -0.5)
        attn_ker = attn_ker.permute(1,0,3,4,2).reshape(B, N_heads, N, M)
        v_ker = v_ker.reshape(1, N_heads, M, ks**2, -1).mean(3) # 1 * N_heads * M * D

        attn_ker = self.attn_drop(attn_ker.softmax(dim=-1))
        x_ker = (attn_ker @ v_ker).transpose(1, 2).reshape(B, N, C)
        x_ker = self.proj(x_ker)
        x_ker = self.proj_drop(x_ker)

        return x_ker, attn_ker


    def forward_kernel_cls(self, x_in, kernel):
        N_heads = self.num_heads
        B, N, C = x_in.shape
        M, ks = kernel.shape[:2]

        qkv = self.qkv(x_in).reshape(B, N, 3, N_heads, C // N_heads).permute(2, 0, 3, 1, 4)
        qkv_ker = self.qkv(kernel).reshape(M, ks*ks, 3, N_heads, C // N_heads).mean(1).permute(1, 2, 0, 3).unsqueeze(1).repeat(1,B,1,1,1)
        q, k_ker, v_ker = qkv[0], qkv_ker[1], qkv_ker[2] # q: B * N_heads * N * D, kv: B * N_heads * M * D

        attn_ker = q @ k_ker.transpose(-1, -2) # B * N_heads * N * M
        attn_ker = attn_ker * (M ** -0.5)
        attn_ker = self.attn_drop(attn_ker.softmax(dim=-1))
        x_ker = (attn_ker @ v_ker).transpose(1, 2).reshape(B, N, C)
        x_ker = self.proj(x_ker)
        x_ker = self.proj_drop(x_ker)

        return x_ker


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_in = norm_layer(dim)
        self.norm_out_patch = norm_layer(2*dim)
        self.norm_out_cls = norm_layer(2*dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(self.ls1(y))
        
        z = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.ls2(z))
        
        if return_attention:
            return x, attn
        return x

    def forward_kernel_patch(self, x_in, kernel):
        x = x_in[:, 1:]
        y, attn_ker = self.attn.forward_kernel_patch(self.norm1(x), \
            self.norm_in(kernel.permute(0,2,3,1)))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = torch.cat([x_in[:, 1:], x], dim=-1)
        x = self.norm_out_patch(x)
        return x, attn_ker
        # x = torch.cat([x_in[:, :1], x], dim=1)
        # return x

    def forward_kernel_cls(self, x_in, kernel):
        x = x_in[:, :1]
        y = self.attn.forward_kernel_cls(self.norm1(x), \
            self.norm_in(kernel.permute(0,2,3,1)))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = torch.cat([x_in[:, :1], x], dim=-1)
        x = self.norm_out_cls(x)
        return x
        # x = torch.cat([x, x_in[:, 1:]], dim=1)
        # return x

    def forward_kernel(self, x_in, kernel, return_attention=False):
        x, attn = self.forward(x_in, return_attention=True)
        x_patch_ker, attn_ker = self.forward_kernel_patch(x, kernel)
        x = torch.cat([
            self.forward_kernel_cls(x, kernel), 
            x_patch_ker
        ], dim=1)

        if return_attention:
            return x, attn, attn_ker
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MemoryBank(nn.Module):
    def __init__(self, num_samples, kernel_size=3):
        super(MemoryBank, self).__init__()
        self._samples = None
        self._kernel_size = kernel_size
        self._num_samples = num_samples
        self._init = False

    @torch.no_grad()
    def update(self, spatial_feat):
        spatial_feat = spatial_feat.data
        bs, d, h, w = spatial_feat.shape
        kernel_size = self._kernel_size

        samples = F.adaptive_avg_pool2d(spatial_feat, (kernel_size, kernel_size))
        n_keep = min(32, len(samples)) if self.ready() else len(samples)
        # n_keep = len(samples)
        keep_idx = torch.randperm(len(samples))[:n_keep]

        if self._samples is None:
            self._samples = samples[keep_idx] # bs * d * ks * ks
        else:
            self._samples = torch.cat([
                samples[keep_idx],
                self._samples[:self._num_samples - len(keep_idx)]
            ])

    # def get_feat(self, spatial_feat):
    #     return F.conv2d(spatial_feat, self.kernel, padding=self._kernel_size//2)

    def ready(self):
        return self._samples is not None and len(self._samples) == self._num_samples

    @property
    def samples(self):
        return self._samples


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768, output_dim=256, hidden_dim=2048,
                 nmb_prototypes=300, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, n_layers_projection_head=3, n_projection_head=1,
                 head_idx_patch=0, head_idx_cls=1, selective_layer_start=11, selective_cache_num=64, selective_kernel_size=3, feat_type_default='all', 
                 init_values=None, feature_regularization='oaf', diffusion_steps=1,
                 diffusion_tau=0.2, diffusion_sigma=1.0, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.input_norm = normalize
        self.head_idx_patch = head_idx_patch
        self.head_idx_cls = head_idx_cls
        assert selective_layer_start == 11
        self.selective_layer_start = selective_layer_start
        valid_regularizers = {'oaf', 'anisotropic_diffusion', 'total_variation', 'none'}
        if feature_regularization not in valid_regularizers:
            raise ValueError(
                'feature_regularization must be one of %s, got %r' %
                (sorted(valid_regularizers), feature_regularization))
        self.feature_regularization = feature_regularization
        self.anisotropic_diffusion = AnisotropicDiffusion(
            num_steps=diffusion_steps,
            tau=diffusion_tau,
            sigma=diffusion_sigma,
        )
        self.cache_samples = MemoryBank(1024, selective_kernel_size)
        self.kernel = nn.Parameter(torch.zeros(selective_cache_num, embed_dim, selective_kernel_size, selective_kernel_size))
        trunc_normal_(self.kernel, std=1)

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])

        # Construct projection head
        if isinstance(nmb_prototypes, int):
            nmb_prototypes = [nmb_prototypes] * n_projection_head

        projection_heads = []
        for i in range(n_projection_head):
            if head_idx_patch == head_idx_cls:
                feat_type = feat_type_default
            elif i == head_idx_patch:
                feat_type = 'patch'
            elif i == head_idx_cls:
                feat_type = 'cls'
            projection_heads.append(Projector(2*embed_dim, hidden_dim[i], output_dim, \
                    nmb_prototypes[i], n_layers_projection_head, feat_type))
        self.projection_heads = nn.ModuleList(projection_heads)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, inputs, masks=None, last_self_attention=False, proj_head=None, update_kernel=False):
        results = {}
        _out = self.forward_backbone(inputs, masks, last_self_attention=last_self_attention, proj_head_idx=proj_head, update_kernel=update_kernel)
        if last_self_attention:
            _out, _attn = _out
            # results['attn'] = torch.cat(_attn)
            results['attn'] = _attn
        results.update(self.forward_head(_out, proj_head))

        return results

    def forward_head(self, x, proj_head_idx=None):
        # Projection with l2-norm bottleneck as prototypes layer is l2-normalized
        results = {}
        # for i, proj_head in enumerate(self.projection_heads):
        if proj_head_idx is not None:
            projection_heads = [self.projection_heads[proj_head_idx]]
            indics = [proj_head_idx]
        else:
            projection_heads = self.projection_heads
            indics = range(len(projection_heads))

        for idx, proj_head in zip(indics, projection_heads):
            out = proj_head(x)
            if idx == self.head_idx_patch:
                if proj_head.feat_type == 'all':
                    out = out[:, 1:]
                if proj_head.feat_type in ['patch', 'all']:
                    bs, res2, d = out.shape
                    h = w = int(math.sqrt(res2))
                    results['feat_%d'%idx] = out.reshape(bs,h,w,d).permute(0,3,1,2)
            if idx == self.head_idx_cls:
                if proj_head.feat_type == 'all':
                    out = out[:, :1]
                if proj_head.feat_type in ['cls', 'all']:
                    results['cls_%d'%idx] = out.squeeze(1)
        return results

    def prepare_tokens(self, x, masks=None):
        x = self.input_norm(x)
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def update_kernel(self, x):
        if self.feature_regularization != 'oaf':
            return
        x = x[:, 1:]
        n = x.shape[0]
        d = x.shape[2]
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(n,h,w,d).permute(0,3,1,2)
        self.cache_samples.update(x)

    @torch.no_grad()
    def cluster(self, pl_module, niter=3000):
        from tqdm import tqdm

        if self.feature_regularization != 'oaf':
            return
        
        if self.cache_samples._init:
            return
        if not self.cache_samples.ready():
            return

        self.cache_samples._init = True

        if pl_module.global_rank == 0:
            samples = self.cache_samples.samples
            N, d, ks = samples.shape[:3]
            M = len(self.kernel)
            samples = samples.reshape(N, d*ks*ks)
            samples = F.normalize(samples, p=2, dim=1)

            center = samples[torch.randperm(N)[:M]]
            center = center.reshape(M, d*ks*ks)
            center = F.normalize(center, p=2, dim=1)
            last_dist = 1e6
            for i in tqdm(range(niter)):
                dist = ((samples[:, None, :] - center[None, :, :])**2).sum(-1)
                min_dist, min_index = dist.min(-1)
                center = torch.stack([samples[[min_index==k]].mean(0) for k in range(M)])
                nan_index = torch.any(torch.isnan(center), dim=1)
                ndead = nan_index.sum().item()
                # print(min_dist.mean(), ndead, N, M)
                center[nan_index] = samples[torch.randperm(N)[:ndead]]
                center = F.normalize(center, p=2, dim=1)
                if last_dist == min_dist.mean().item():
                    break
                last_dist = min_dist.mean()

            self.kernel.data = center.reshape(M, d, ks, ks)

        self.kernel.data = pl_module.broadcast(self.kernel.data, 0)

    def forward_regularized(self, block, x, return_attention=False):
        """Run the last ViT block and replace its OAF branch."""
        x, attn = block(x, return_attention=True)
        patch = x[:, 1:]
        batch_size, num_patches, dim = patch.shape
        height = int(math.sqrt(num_patches))
        if height * height != num_patches:
            raise ValueError('anisotropic diffusion requires a square patch grid')

        if self.feature_regularization == 'anisotropic_diffusion':
            patch_map = patch.transpose(1, 2).reshape(batch_size, dim, height, height)
            filtered_patch = self.anisotropic_diffusion(patch_map)
            filtered_patch = filtered_patch.flatten(2).transpose(1, 2)
        else:
            # TV is applied as an auxiliary loss to the projected patch map.
            # Duplicating the branch retains OAF's 2D projector/checkpoint shape.
            filtered_patch = patch

        patch = block.norm_out_patch(torch.cat([patch, filtered_patch], dim=-1))
        cls = block.norm_out_cls(torch.cat([x[:, :1], x[:, :1]], dim=-1))
        x = torch.cat([cls, patch], dim=1)
        if return_attention:
            return x, attn
        return x

    def forward_backbone(self, x, masks=None, last_self_attention=False, proj_head_idx=None, update_kernel=False):
        x = self.prepare_tokens(x, masks)

        for i, blk in enumerate(self.blocks[:self.selective_layer_start]):
            x = blk(x)

        if update_kernel:
            self.update_kernel(x)
        for i, blk in enumerate(self.blocks[self.selective_layer_start:-1]):
            if self.feature_regularization == 'oaf':
                x = blk.forward_kernel(x, self.kernel.data)
            else:
                x = self.forward_regularized(blk, x)
        if self.feature_regularization == 'oaf':
            x = self.blocks[-1].forward_kernel(
                x, self.kernel.data, return_attention=last_self_attention)
        else:
            x = self.forward_regularized(
                self.blocks[-1], x, return_attention=last_self_attention)
        if last_self_attention:
            x, attn = x[:2]
            attn = attn[:, :, 0, 1:]
            return x, attn
        return x

    def get_last_feature(self, x):
        x = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks[:self.selective_layer_start]):
            x = blk(x)

        for i, blk in enumerate(self.blocks[self.selective_layer_start:]):
            if self.feature_regularization == 'oaf':
                x = blk.forward_kernel(x, self.kernel.data)
            else:
                x = self.forward_regularized(blk, x)

        x_cls = x[:, 0]
        x_patch = x[:, 1:]
        bs, res2, d = x_patch.shape
        h = w = int(math.sqrt(res2))
        x_patch = x_patch.reshape(bs,h,w,d).permute(0,3,1,2)
        return x_cls, x_patch

    def get_middle_feature(self, x):
        x = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks[:self.selective_layer_start]):
            x = blk(x)

        for i, blk in enumerate(self.blocks[self.selective_layer_start:]):
            if self.feature_regularization == 'oaf':
                x, attn, attn_ker = blk.forward_kernel(
                    x, self.kernel.data, return_attention=True)
            else:
                x, attn = self.forward_regularized(blk, x, return_attention=True)
                attn_ker = None

        x_cls = x[:, 0]
        x_patch = x[:, 1:]
        bs, res2, d = x_patch.shape
        h = w = int(math.sqrt(res2))
        x_patch = x_patch.reshape(bs,h,w,d).permute(0,3,1,2)
        return x_cls, x_patch, attn, attn_ker



class Projector(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_prototypes, nlayers, feat_type) -> None:
        super().__init__()
        self.feat_type = feat_type

        self.nlayers = nlayers
        layers = [nn.Linear(embed_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.projection_head = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(output_dim, num_prototypes, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        if self.nlayers == 0:
            return x
        bs, n_patch, d = x.shape
        x = self.projection_head(x.view(-1, d))
        x = x.view(bs, n_patch, -1)
        if self.feat_type == 'cls':
            x = x[:, :1]
        elif self.feat_type == 'patch':
            x = x[:, 1:]
        x = F.normalize(x.reshape(-1, x.shape[-1]), dim=1, p=2)
        x = self.last_layer(x)
        return x.view(bs, -1, x.shape[-1])


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class NormalizeConv(nn.Module):
    def __init__(self,in_channel,normal_mean=[0.485, 0.456, 0.406],normal_std=[0.229, 0.224, 0.225]):
        super().__init__()

        normal_mean = [-i/std for i,std in zip(normal_mean,normal_std)]
        normal_std = [1/std for std in normal_std]
        weight = np.array([[[normal_std]]])
        bias = np.array(normal_mean)
        weight = weight.transpose((3,0,1,2))
        self.conv = nn.Conv2d(in_channel,in_channel,1,stride=1,padding=0,groups=in_channel,bias=True)
        for i in self.conv.parameters():
            i.requires_grad = False
        self.conv.weight = nn.Parameter(torch.Tensor(weight),requires_grad=False)
        self.conv.bias = nn.Parameter(torch.Tensor(bias),requires_grad=False)

    def forward(self,x):
        out = self.conv(x)
        return out

def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
