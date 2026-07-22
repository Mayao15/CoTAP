import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
import wget
import math

from os.path import join
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchvision import models
from torchvision.transforms import GaussianBlur
from torch.utils.tensorboard.summary import hparams
from easydict import EasyDict as edict


@torch.jit.script
def shuffle(x):
    return x[torch.randperm(x.shape[0])]

def norm(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-10)

def add_hparams_fixed(writer, hparam_dict, metric_dict, global_step):
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_step)

def share_cfg(cfg):
    if not 'share' in cfg.loss_fn.keys():
        cfg.loss_fn.share = dict()
    cfg.loss_fn.share.dim = cfg.model.dim
    cfg.loss_fn.share.head_idx_patch = cfg.model.head_idx_patch
    cfg.loss_fn.share.head_idx_cls = cfg.model.head_idx_cls
    cfg.loss_fn.share.nmb_crops = cfg.dataset_train.nmb_crops
    cfg.loss_fn.share.world_size = cfg.training.num_gpus
    cfg.loss_fn.share.feature_regularization = getattr(
        cfg.model, 'feature_regularization', 'oaf')
    if cfg.model.arch == 'dino':
        cfg.loss_fn.share.patch_size = cfg.model.dino_patch_size
        cfg.dataset_train.patch_size = cfg.model.dino_patch_size
    else:
        cfg.loss_fn.share.patch_size = 16
    for key in cfg.dataset_val.keys():
        cfg.dataset_val[key].dim = cfg.model.dim

    return cfg

def rename_blocks_param(src_state_dict, dst_state_dict, vit_type):
    if vit_type == "dinov2":
        return src_state_dict
    elif vit_type in ['splitattn', 'cross_stitch']:
        return rename_blocks_param_splitattn(src_state_dict, dst_state_dict)
    else:
        return rename_blocks_param_default(src_state_dict, dst_state_dict)

def rename_blocks_param_default(src_state_dict, dst_state_dict):
    depth_in, depth_out, depth_share, n_heads = 0, 0, 0, 1
    for k in dst_state_dict.keys():
        if k.startswith('blocks_in.'):
            n_heads = max(n_heads, int(k.split('.')[1]) + 1)
            depth_in = max(depth_in, int(k.split('.')[2]) + 1)
        if k.startswith('blocks_out.'):
            n_heads = max(n_heads, int(k.split('.')[1]) + 1)
            depth_out = max(depth_out, int(k.split('.')[2]) + 1)
        if k.startswith('blocks.'):
            depth_share = max(depth_share, int(k.split('.')[1]) + 1)

    state_dict = {}
    for k in src_state_dict.keys():
        if k.startswith('blocks.'):
            index = int(k.split('.')[1])
            for c in range(n_heads):
                if index < depth_in:
                    rename = '.'.join(['blocks_in.%d.%d'%(c, index)] + k.split('.')[2:])
                elif index < depth_share + depth_in:
                    rename = '.'.join(['blocks.%d'%(index-depth_in)] + k.split('.')[2:])
                else:
                    rename = '.'.join(['blocks_out.%d.%d'%(c, index-depth_in-depth_share)] + k.split('.')[2:])
                state_dict[rename] = src_state_dict[k]
        else:
            state_dict[k] = src_state_dict[k]

    return state_dict
    
def rename_blocks_param_splitattn(src_state_dict, dst_state_dict):
    depth_in, depth_out, depth_share, n_heads = 0, 0, 0, 1
    for k in dst_state_dict.keys():
        if k.startswith('blocks_in.'):
            n_heads = max(n_heads, int(k.split('.')[1]) + 1)
            depth_in = max(depth_in, int(k.split('.')[2]) + 1)
        if k.startswith('blocks_out.'):
            # n_heads = max(n_heads, int(k.split('.')[1]) + 1)
            # depth_out = max(depth_out, int(k.split('.')[2]) + 1)
            if 'attn' in k:
                n_heads = max(n_heads, int(k.split('.')[3]) + 1)
            depth_out = max(depth_out, int(k.split('.')[1]) + 1)
        if k.startswith('blocks.'):
            depth_share = max(depth_share, int(k.split('.')[1]) + 1)

    state_dict = {}
    for k in src_state_dict.keys():
        if k.startswith('blocks.'):
            index = int(k.split('.')[1])
            for c in range(n_heads):
                if index < depth_in:
                    rename = '.'.join(['blocks_in.%d.%d'%(c, index)] + k.split('.')[2:])
                elif index < depth_share + depth_in:
                    rename = '.'.join(['blocks.%d'%(index-depth_in)] + k.split('.')[2:])
                else:
                    if 'attn' in k:
                        rename = '.'.join(['blocks_out.%d.attn.%d'%(index-depth_in-depth_share, c)] + k.replace('attn.', '').split('.')[2:])
                    else:
                        rename = '.'.join(['blocks_out.%d'%(index-depth_in-depth_share)] + k.split('.')[2:])
                # print(k, index, rename)
                state_dict[rename] = src_state_dict[k]
        else:
            # print(k)
            state_dict[k] = src_state_dict[k]

    return state_dict


def load_checkpoint(model, ckpt_path, vit_type, load_teacher=False):
    if ckpt_path is None:
        print("Warning: ckpt_path is None, skipping checkpoint loading.")
        return model
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    if "teacher" in state_dict.keys():
        state_dict = state_dict["teacher"]

    # DINOv2 keys are already clean, no prefix removal needed usually
    # But check if we need to remove specific prefixes
    
    if vit_type == "dinov2":
        # DINOv2 checkpoint might be clean or have 'module.' if saved from DDP
        # Based on your inspection, it seems clean: ['cls_token', 'pos_embed', ...]
        pass
    else:
        remove_prefix = ['net_teacher', 'teacher', 'model_tea']
        change_prefix = ['net', 'model', 'backbone']
    if load_teacher:
        remove_prefix[0] = 'net'
        change_prefix[0] = 'net_teacher'

    for rp in remove_prefix:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(rp + '.')}

    for cp in change_prefix:
        state_dict = {k.replace(cp + '.', ''): v for k, v in state_dict.items()}

    # state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    # state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('net_teacher.')}
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('teacher.')}
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model_tea.')}
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.projection_heads')}

    if vit_type != "dinov2":
        state_dict = rename_blocks_param(state_dict, model.model.state_dict(), vit_type)
        
    # Resize pos_embed for DINOv2 if shape mismatch
    if 'pos_embed' in state_dict and model.model.pos_embed.shape != state_dict['pos_embed'].shape:
        print(f"Resizing pos_embed from {state_dict['pos_embed'].shape} to {model.model.pos_embed.shape}")
        pos_embed_old = state_dict['pos_embed']
        pos_embed_new = model.model.pos_embed
        
        # Determine number of extra tokens (CLS, registers, etc.)
        # DINOv2 pretrained usually has 1 CLS token (vits14) or CLS + 4 registers (vits14_reg)
        # But here target model has 257 tokens (1 CLS + 256 patches)
        # Source checkpoint has 1370 tokens (1 CLS + 1369 patches -> 37x37 patches -> 518x518 img)
        
        # Assuming CLS token is at index 0 and registers (if any) follow.
        # But standard vits14 pretrain only has CLS.
        num_extra_tokens = 1 
        
        # If registers are present in checkpoint but not in model, or vice-versa, we need to handle that.
        # For now, assume simple resize of patch tokens.
        
        cls_tok = pos_embed_old[:, :num_extra_tokens]
        pos_tokens = pos_embed_old[:, num_extra_tokens:]
        
        N_old = pos_tokens.shape[1]
        N_new = pos_embed_new.shape[1] - num_extra_tokens
        
        gs_old = int(math.sqrt(N_old))
        gs_new = int(math.sqrt(N_new))
        
        if gs_old * gs_old != N_old:
             print(f"Warning: Old pos_embed patch count {N_old} is not a perfect square. Interpolation might be wrong.")
        
        pos_tokens = pos_tokens.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, N_new, -1)
        
        state_dict['pos_embed'] = torch.cat((cls_tok, pos_tokens), dim=1)

    msg = model.model.load_state_dict(state_dict, strict=False)
    print('\nPretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))
    return model

@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


def one_hot_feats(labels, n_classes):
    return F.one_hot(labels, n_classes).permute(0, 3, 1, 2).to(torch.float32)


def load_model(model_type, data_dir):
    if model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download",
                          model_file)
        model_weights = torch.load(model_file)
        # model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
        #                          'model' in name}
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    model.eval()
    model.cuda()
    return model


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) >= 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
            shape = [len(batch)] + list(elem.shape)
            out = out.reshape(shape)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def cosine_scheduler(base_value, final_value, total_iter, warmup_iter=0, start_warmup_value=1, pad=0):

    iters = np.arange(total_iter - warmup_iter)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    if warmup_iter > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iter)
        schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == total_iter
    if len(schedule) < pad:
        schedule = np.concatenate((schedule, schedule[-1] * np.ones(pad - len(schedule))))

    return schedule


def process_attentions(attentions: torch.Tensor, spatial_res: int=14, threshold: float = 0.6, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # # Blur attentions
    attentions = attentions.reshape(attentions.size(0), -1, spatial_res, spatial_res)
    attentions = attentions.mean(1, keepdim=True)
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    # attentions /= (attentions.view(-1, spatial_res**2).max(1)[0]).reshape(-1, 1, 1, 1)

    # return attentions

    attentions = attentions.reshape(attentions.size(0), 1, spatial_res**2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()

    # # Remove components with less than 3 pixels
    # for j, th_att in enumerate(th_attn):
    #     labelled = label(th_att.cpu().numpy())
    #     for k in range(1, np.max(labelled) + 1):
    #         mask = labelled == k
    #         if np.sum(mask) <= 2:
    #             th_attn[j, 0][mask] = 0

    return th_attn.detach()
