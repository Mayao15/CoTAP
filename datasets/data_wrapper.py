import os
import os.path as osp
from posixpath import split
import random

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .transforms import get_transform, LeopartTransforms
from .ade20k import ADE20k
from .cityscapes import CityscapesSeg
from .coco import COCO
from .imagenet import ImageNet
from .voc import VOCDataset


class DataWrapper(Dataset):
    def __init__(self,
                 cfg,
                 image_set,
                 label=False,
                 pos_labels=False,
                 pos_images=False,
                 randomaug=False,
                 use_leopart_transform=False,
                 num_neighbors_chosen=1,
                 res=None,
                 use_mask=False,
                 random_crop_res=-1,
                 ):
        super(DataWrapper).__init__()

        self.cfg = cfg
        self.num_neighbors_chosen = num_neighbors_chosen
        self.num_neighbors = cfg.num_neighbors
        self.dataset_name = cfg.dataset_name
        self.image_set = image_set
        self.label = label
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.randomaug = randomaug
        self.use_leopart_transform = use_leopart_transform
        self.lmdb_path = self.cfg.lmdb_path
        self.res = res
        self.use_mask = use_mask
        self.random_crop_res = random_crop_res

        if use_mask:
            from .masking import MaskingGenerator

            if use_leopart_transform:
                res_all = [cfg.size_crops[0]] * cfg.nmb_crops[0] \
                    + [cfg.size_crops[1]] * cfg.nmb_crops[1]
                self.mask_generator = [MaskingGenerator(
                    res // cfg.patch_size,
                    int(0.5 * (res // cfg.patch_size)**2),
                ) for res in res_all]
            else:
                self.mask_generator = MaskingGenerator(
                    cfg.res // cfg.patch_size,
                    int(0.5 * (cfg.res // cfg.patch_size)**2),
                )

        self.build_dataset()

    def __len__(self):
        return max([len(ds) for ds in self.dataset])

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def build_transform(self):
        cfg = self.cfg
        if self.res is not None:
            res = self.res
        else:
            res = self.cfg.res

        if self.use_leopart_transform:
            transform = get_transform(
                res, False, self.image_set, self.randomaug, True,
                cfg.size_crops, cfg.nmb_crops, cfg.min_scale_crops, cfg.max_scale_crops
            )
        else:
            transform = get_transform(
                res, False, self.image_set, self.randomaug, False, res_resize=self.cfg.get('res_resize', None)
            )
        if 'target_res' not in self.cfg.keys():
            self.cfg.target_res = res
        target_transform=get_transform(self.cfg.target_res, True, self.image_set, False)

        return transform, target_transform

    def reset_res(self, res):
        if res is None or self.cfg.res == res:
            return
        self.cfg.res = res
        self.cfg.size_crops[0] = res
        self.build_dataset()

    def build_dataset(self):
        cfg = self.cfg
        dataset_name_list = cfg.dataset_name.split('+')
        self.dataset = []
        for dataset_name in dataset_name_list:
            extra_args = dict(lmdb_path=osp.join(self.lmdb_path, '%s_%s.lmdb'%(
                    dataset_name, self.image_set)) if self.lmdb_path is not None else None)
            if dataset_name == "imagenet100":
                self.n_classes = 100
                dataset_class = ImageNet
                extra_args.update(dict(coarse_labels=False, num_classes=100))
            elif dataset_name == "imagenet1k":
                self.n_classes = 1000
                dataset_class = ImageNet
                extra_args.update(dict(coarse_labels=False, num_classes=1000, full_data=self.cfg.full_data))
            elif dataset_name == 'voc':
                self.n_classes = 21
                dataset_class = VOCDataset
                extra_args.update(dict(random_crop_res=self.random_crop_res))
            elif dataset_name == 'ade20k':
                self.n_classes = 150
                dataset_class = ADE20k
                extra_args.update(dict(random_crop_res=self.random_crop_res))
            elif dataset_name == "cityscapes":
                self.n_classes = 19
                dataset_class = CityscapesSeg  
                extra_args.update(dict(random_crop_res=self.random_crop_res))
            elif dataset_name == "cocostuff27-thing":
                self.n_classes = 12
                dataset_class = COCO
                extra_args.update(dict(coarse_labels=False, subset=0, exclude_things=False, exclude_stuff=True))
            elif dataset_name == "cocostuff27":
                self.n_classes = 27
                dataset_class = COCO
                extra_args.update(dict(random_crop_res=self.random_crop_res, coarse_labels=False, subset=0, exclude_things=False))
                if self.image_set == "val":
                    extra_args["subset"] = 7
            elif dataset_name == "coco":
                self.n_classes = 27
                dataset_class = COCO
                extra_args.update(dict(random_crop_res=-1, coarse_labels=False, subset=None, exclude_things=False))
            else:
                raise ValueError("Unknown dataset: {}".format(dataset_name))

            transform, target_transform = self.build_transform()
            dataset = dataset_class(
                root=cfg.pytorch_data_dir,
                image_set=self.image_set,
                transform=transform,
                target_transform=target_transform,
                **extra_args)

            nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
            feature_cache_file = osp.join(cfg.pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            'vit_base', nice_dataset_name, self.image_set, None, 224))

            if self.pos_labels or self.pos_images:
                if not os.path.exists(feature_cache_file):
                    raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
                else:
                    loaded = np.load(feature_cache_file)
                    dataset.nns = loaded["nns"]

                assert len(dataset) == dataset.nns.shape[0]

            self.dataset.append(dataset)

    def _getitem_single_dataset(self, ind, dataset):
        ind %= len(dataset)
        pack = dataset[ind]

        if self.pos_images or self.pos_labels:
            perm = torch.randperm(self.num_neighbors)
            pos_idx = perm[:self.num_neighbors_chosen]
            pack_pos = dataset[dataset.nns[ind][pos_idx[0]]]
        else:
            pack_pos = None

        if self.use_leopart_transform:
            img, bbox_dict = pack[0]
            ret = {
                "ind": ind,
                "img": img,
                "bbox_dict": bbox_dict,
            }

            if pack_pos is not None:
                img_pos, bbox_dict_pos = pack_pos[0]
                ret.update({
                    "img_pos": img_pos,
                    "bbox_dict_pos": bbox_dict_pos
                })
        else:
            ret = {
                "ind": ind,
                "img": pack[0],
            }

            if pack_pos is not None:
                ret.update({
                    "img_pos": pack_pos[0],
                })

        if self.use_mask:
            if torch.rand(1) > 0.5:
                n_tokens = (self.cfg.res // self.cfg.patch_size)**2
                N = int(n_tokens * random.uniform(0.1, 0.5))
            else:
                N = 0

            ret['mask'] = [torch.BoolTensor(mg(N)) for mg in self.mask_generator]
            if pack_pos is not None:
                ret['mask_pos'] = [torch.BoolTensor(mg(N)) for mg in self.mask_generator]

        if self.label:
            ret["label"] = pack[1]

        if self.pos_labels:
            ret["label_pos"] = pack_pos[1]

        return ret

    def __getitem__(self, index):
        return [self._getitem_single_dataset(index, dataset) for dataset in self.dataset]
