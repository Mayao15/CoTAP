import faiss
import timm
import numpy as np
import os
import time
import torch
import torch.nn as nn

from collections import defaultdict
from functools import partial
from skimage.measure import label
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchvision.transforms import GaussianBlur
from torchvision import models
from typing import Optional, List, Tuple, Dict


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """
    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(self, is_global_zero: bool, many_to_one: bool = False,
                precision_based: bool = False, linear_probe : bool = False) -> Tuple[float, List[np.int64],
                                                                                     List[np.int64], List[np.int64],
                                                                                     List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(gt, pred, self.num_pred_classes, self.num_gt_classes, many_to_one=many_to_one,
                                     precision_based=precision_based, linear_probe=linear_probe)

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt:int,
                     many_to_one=False, precision_based=False, linear_probe=False) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64],
                                                  List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
            matched_bg_clusters = {}
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
                matched_bg_clusters = len(match[0]) / num_pred
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0
                matched_bg_clusters = 1/num_gt

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == i_part)
            tmp_pred = (reordered_preds == i_part)
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print("IoUs computed")
        return np.mean(jac), tp, fp, fn, reordered_preds.astype(int).tolist(), matched_bg_clusters

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        print("Parallelizing iou computation")
        start = time.time()
        score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
                                                 for c2 in range(num_pred) for c1 in range(num_gt))
        print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray,
                                                                                                      np.ndarray]:
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k,v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        print("matched clusters to gt classes:")
        return gt_to_matches

def get_backbone_weights(arch: str, method: str, patch_size: int = None,
                         weight_prefix: Optional[str]= "model", ckpt_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Load backbone weights into formatted state dict given arch, method and patch size as identifiers.
    :param arch: Target architecture. Currently supports resnet50, vit-small and vit-base.
    :param method: Method identifier.
    :param patch_size: Patch size of ViT. Ignored if arch is not ViT.
    :param weight_prefix: Optional prefix of weights to match model naming.
    :param ckpt_path: Optional path to checkpoint containing state_dict to be processed.
    :return: Dictionary mapping to weight Tensors.
    """
    def identity_transform(x): return x
    arch_to_args = {
        'vit-small16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
                             torch.hub.load_state_dict_from_url,
                             lambda x: x["teacher"]),
        'vit-small16-ours': (ckpt_path,
                             partial(torch.load, map_location=torch.device('cpu')),
                             lambda x: {k: v for k, v in x.items() if k.startswith('model')}),
        'vit-small16-mocov3': ("https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                               torch.hub.load_state_dict_from_url,
                               lambda x: {k: v for k, v in x.items() if k.startswith('module.base_encoder')}),
        'vit-small16-sup_vit': ('vit_small_patch16_224',
                            lambda x: timm.create_model(x,  pretrained=True).state_dict(),
                            identity_transform),
        'vit-base16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth",
                            torch.hub.load_state_dict_from_url,
                            lambda x: x["teacher"]),
        'vit-base8-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["teacher"]),
        'vit-base16-mae': ("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["model"]),
        'resnet50-sup_resnet': ("",
                                lambda x: models.resnet50(pretrained=True).state_dict(),
                                identity_transform),
        'resnet50-maskcontrast': (ckpt_path,
                                  torch.load,
                                  lambda  x: x["model"]),
        'resnet50-swav': ("https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-moco': ("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-densecl': (ckpt_path,
                             torch.load,
                             identity_transform),
    }
    arch_to_args['vit-base16-ours'] = arch_to_args['vit-base8-ours'] = arch_to_args['vit-small16-ours']

    if "vit" in arch:
        url, loader, weight_transform = arch_to_args[f"{arch}{patch_size}-{method}"]
    else:
        url, loader, weight_transform = arch_to_args[f"{arch}-{method}"]
    weights = loader(url)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = weight_transform(weights)
    prefix_idx, prefix = get_backbone_prefix(weights, arch)
    if weight_prefix:
        return {f"{weight_prefix}.{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
                and "head" not in k and "prototypes" not in k}
    return {f"{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
            and "head" not in k and "prototypes" not in k}


def get_backbone_prefix(weights: Dict[str, torch.Tensor], arch: str) -> Optional[Tuple[int, str]]:
    # Determine weight prefix if returns empty string as prefix if not existent.
    if 'vit' in arch:
        search_suffix = 'cls_token'
    elif 'resnet' in arch:
        search_suffix = 'conv1.weight'
    else:
        raise ValueError()
    for k in weights:
        if k.endswith(search_suffix):
            prefix_idx = len(k) - len(search_suffix)
            return prefix_idx, k[:prefix_idx]
