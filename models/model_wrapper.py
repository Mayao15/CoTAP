import torch
import torch.nn.functional as F
import time
import math

from utils import *


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg, require_grad=False, load_pretrain=True, pretrained_weights=None, enable_branch=-1):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        vit_type = self.cfg.vit_type
        self.v0 = None
        self.v1 = None

        ####################
        if vit_type == 'splitattn':
            from .vit import vision_transformer_splitattn as vits
        elif vit_type == 'cross_stitch':
            from .vit import vision_transformer_cross_stitch as vits
        elif vit_type == 'skip':
            from .vit import vision_transformer_skip as vits
        elif vit_type == 'depth':
            from .vit import vision_transformer_depth as vits
        elif vit_type == 'default':
            from .vit import vision_transformer as vits
        elif vit_type == "dinov2":
            from .vit import vision_transformer_v2 as vits
        else:
            raise NotImplementedError

        if arch == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 14:
            url = "dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        # DINOv2 uses a different init signature than standard ViT
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            drop_path_uniform=False,
            output_dim=cfg.output_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers_projection_head=cfg.n_layers_projection_head,
            nmb_prototypes=cfg.nmb_prototypes,
            n_projection_head=cfg.n_projection_head,
            head_idx_patch=cfg.head_idx_patch,
            head_idx_cls=cfg.head_idx_cls,
            depth_in=cfg.depth_in,
            depth_out=cfg.depth_out,
            selective_layer_start=cfg.selective_layer_start,
            selective_cache_num=cfg.selective_cache_num,
            selective_kernel_size=cfg.selective_kernel_size,
            feat_type_default=cfg.feat_type_default,
            init_values=getattr(cfg, 'init_values', None),
        )
            
        if not require_grad:
            for p in self.model.parameters():
                p.requires_grad = False

        if enable_branch == 0:
            for p in self.model.projection_heads[1].parameters():
                p.requires_grad = False
            for p in self.model.blocks_out[1].parameters():
                p.requires_grad = False
        elif enable_branch == 1:
            for p in self.model.projection_heads[0].parameters():
                p.requires_grad = False
            for p in self.model.blocks_out[0].parameters():
                p.requires_grad = False

        if load_pretrain:
            if pretrained_weights is not None:
                load_checkpoint(self, pretrained_weights, vit_type)
            else:
                print("\nSince no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/" + url, map_location='cpu')
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.projection_heads')}
                
                # Resize pos_embed if needed (e.g. DINOv2 518x518 -> 224x224)
                if 'pos_embed' in state_dict:
                    pos_embed_old = state_dict['pos_embed']
                    pos_embed_new = self.model.pos_embed
                    if pos_embed_old.shape != pos_embed_new.shape:
                        print(f"Resizing pos_embed from {pos_embed_old.shape} to {pos_embed_new.shape}")
                        
                        # Handle potential registers (DINOv2 with registers has them after CLS)
                        # But standard DINOv2 pretrain (vits14) has no registers.
                        # Assuming [CLS, PatchTokens...] structure.
                        num_extra_tokens = 1 # CLS token
                        
                        # Check if old and new have same embedding dim
                        if pos_embed_old.shape[-1] != pos_embed_new.shape[-1]:
                            print("Warning: Embedding dimension mismatch in pos_embed!")
                        
                        cls_tok = pos_embed_old[:, :num_extra_tokens]
                        pos_tokens = pos_embed_old[:, num_extra_tokens:]
                        
                        N_old = pos_tokens.shape[1]
                        N_new = pos_embed_new.shape[1] - num_extra_tokens
                        
                        gs_old = int(math.sqrt(N_old))
                        gs_new = int(math.sqrt(N_new))
                        
                        if gs_old * gs_old != N_old:
                             print(f"Error: Old pos_embed patch count {N_old} is not a perfect square.")
                        
                        pos_tokens = pos_tokens.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                        pos_tokens = F.interpolate(pos_tokens, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
                        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, N_new, -1)
                        
                        state_dict['pos_embed'] = torch.cat((cls_tok, pos_tokens), dim=1)
                
                # Handle keys mismatch between DINOv2 and expected keys
                # DINOv2: blocks.0.attn.qkv.weight
                # Expected (maybe): blocks.0.attn.qkv.weight (seems matching)
                # But check for other differences
                
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(msg)

    def setup_grad_norm(self):
        if self.v0 is not None:
            return
        self.v0 = []
        self.v1 = []
        for param in self.parameters():
            self.v0.append(1 + 0*param.clone())
            self.v1.append(1 + 0*param.clone())
        self.v0 = nn.ParameterList(self.v0)
        self.v1 = nn.ParameterList(self.v1)

    def process(self, img, outputs, head_idx=0):
        bs = img.shape[0]
        if 'feat_%d'%head_idx in outputs.keys():
            h = w = int(img.shape[-1] // self.patch_size)
            return {'feat_%d'%head_idx: \
                outputs['feat_%d'%head_idx].reshape(bs, h, w, -1).permute(0,3,1,2)}
        else:
            return {'cls_%d'%head_idx: \
                outputs['cls_%d'%head_idx].reshape(bs, -1)}

    def forward(self, img, masks=None, return_attn=False, proj_head=None, update_kernel=False):

        outputs = self.model(img, masks, last_self_attention=return_attn, \
            proj_head=proj_head, update_kernel=update_kernel)

        if return_attn:
            bs = img.shape[0]
            h = w = int(img.shape[-1] // self.patch_size)
            attn_flatten = outputs['attn']
            attn_soft = attn_flatten.reshape(bs, attn_flatten.shape[1], h, w)
            attn = process_attentions(attn_soft, spatial_res=h)
        else:
            attn = None
        outputs['attn'] = attn

        return outputs

    def get_test_features(self, img, return_attn=False):
        feats_cls, feats_patch = self.model.get_last_feature(img)
        return feats_cls, feats_patch

    def get_test_cls_features(self, img):
        feats = self.model.get_last_feature(img)
        n = feats.shape[0]
        h = w = int((feats.shape[1] - 1)**0.5)
        feats = feats[:,0]
        return feats

    @torch.no_grad()
    def ema_update(self, model, m, beta=1e-3):
        if not isinstance(model, list):
            for param_q, param_k in zip(model.parameters(), self.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        elif len(model) == 2:
            self.setup_grad_norm()

            for idx, (param_q0, param_q1, param_k) in enumerate(zip(model[0].parameters(), model[1].parameters(), self.parameters())):
                # if param_k.requires_grad is False:
                if param_q0.requires_grad is False:
                    param_k.data.mul_(m[1]).add_((1 - m[1]) * param_q1.detach().data)
                elif param_q1.requires_grad is False:
                    param_k.data.mul_(m[0]).add_((1 - m[0]) * param_q0.detach().data)
                else:
                    param_k.data.mul_(m[0]*m[1]).add_(
                        (1+m[1])/2*(1 - m[0]) * param_q0.detach().data \
                            + (1+m[0])/2*(1 - m[1]) * param_q1.detach().data
                    )
        else:
            raise NotImplementedError

class ResNet(nn.Module):

    def __init__(self, dim, cfg, require_grad=False):
        super().__init__()
        from .resnet import resnet50

        self.cfg = cfg
        self.dim = dim
        self.model = resnet50(head_type='early_return')

        if not require_grad:
            for p in self.model.parameters():
                p.requires_grad = False
        
        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["state_dict"]
            # remove `module.` prefix
            state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('net_teacher.')}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('teacher.')}
            state_dict = {k: v for k, v in state_dict.items() if not k.endswith('num_batches_tracked')}
            msg = self.load_state_dict(state_dict, strict=False)
            print('\nPretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            assert False

    def forward(self, img, head_index=0):
        self.model.eval()
        with torch.no_grad():
            image_feat = self.model(img, head_index=0)

        return image_feat

    def get_test_features(self, img, return_attn=False):
        feats = self.forward(img)
        if return_attn:
            return feats, None

        return feats

def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)
