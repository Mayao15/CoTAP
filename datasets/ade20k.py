import os
import os.path as osp
import lmdb
import pickle
import numpy as np
import torch
import torch.multiprocessing

from PIL import Image
from torch.utils.data import Dataset

from .utils import build_lmdb
from .cityscapes import RandomCrop


class ADE20k(Dataset):

    def __init__(self, root, image_set, transform, target_transform, random_crop_res, lmdb_path=None):
        super(ADE20k, self).__init__()

        self.split = image_set
        self.root = osp.join(root, "ADEChallengeData2016")
        self.transform = transform
        self.label_transform = target_transform
        self.lmdb_path = lmdb_path
        self.lmdb_txn = None
        self.random_crop = RandomCrop(random_crop_res, (0.64, 1))

        assert self.split in ["train", "val"]

        img_root = osp.join(self.root, 'images', image_set)
        self.image_files = [osp.join(img_root, i) for i in sorted(os.listdir(img_root))]
        lbl_root = osp.join(self.root, 'annotations', image_set)
        self.label_files = [osp.join(lbl_root, i) for i in sorted(os.listdir(lbl_root))]

    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        if self.lmdb_path is not None:
            if self.lmdb_txn is None:
                lmdb_dir = self.lmdb_path
                build_lmdb(lmdb_dir, self.image_files)
                env = lmdb.open(lmdb_dir, map_size=int(1e12), readonly=True, lock=False, readahead=True, meminit=False)
                self.lmdb_txn = env.begin(write=False)
                self.meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), "rb"))

            img_buff = self.lmdb_txn.get(image_path.encode('ascii'))
            C, H, W = [int(i) for i in self.meta_info[image_path].split('_')]
            img = np.frombuffer(img_buff, dtype=np.uint8)
            img = torch.from_numpy(img.copy())
            img = img.float().reshape(C, H, W)
            img = img.div(255)
        else:
            img = Image.open(image_path).convert("RGB")
            label_path = self.label_files[index]
            label = Image.open(label_path)
            img, label = self.random_crop(img, label)

            img = np.array(img).astype(dtype=np.uint8)
            img = torch.from_numpy(img).float().permute(2,0,1)
            img = img.div(255)

        img = self.transform(img)

        label = self.label_transform(label).squeeze(0)
        label -= 1
        label[label == -1] = 255

        return img, label

    def __len__(self) -> int:
        return len(self.image_files)