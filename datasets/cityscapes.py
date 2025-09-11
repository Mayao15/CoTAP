import os
import os.path as osp
import lmdb
import pickle
import numpy as np
import torch
import torch.multiprocessing

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision import transforms

from .utils import build_lmdb


class RandomCrop(object):
    def __init__(self, res, scale, ratio=(3.0/4.0, 4.0/3.0)) -> None:
        if isinstance(res, int):
            res = (res, res)
        self.res = res
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, label):
        if self.res[0] <= 0:
            return image, label
        
        if np.random.rand() > 0.5:
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = torch.flip(image, [-1])
            if isinstance(label, Image.Image):
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                label = torch.flip(label, [-1])

        params = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = transforms.functional.resized_crop(image, *params, size=self.res)
        label = transforms.functional.resized_crop(label, *params, size=self.res, interpolation=Image.NEAREST)
        return image, label


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform, random_crop_res=-1, lmdb_path=None):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.random_crop = RandomCrop(random_crop_res, (0.36, 1))

        self.lmdb_path = lmdb_path
        self.root = osp.join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)

        self.transform = transform
        self.target_transform = target_transform
        self.void_id = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
        self.id_map = [-1 for i in range(34)]
        cnt_ok = 0

        for i in range(34):
            if i in self.void_id:
                self.id_map[i] = 255
            else:
                self.id_map[i] = cnt_ok
                cnt_ok += 1

    def __getitem__(self, index):
        image, target = self.inner_loader[index]
        if self.transform is not None:
            image = np.array(image).astype(dtype=np.uint8)
            image = torch.from_numpy(image).float().permute(2,0,1)
            image = image.div(255)

            image, target = self.random_crop(image, target)

            image = self.transform(image)

            target = self.target_transform(target)
            for i in range(len(self.id_map)):
                target[target == i] = self.id_map[i]

        return image, target.squeeze(0)

    def __len__(self):
        return len(self.inner_loader)