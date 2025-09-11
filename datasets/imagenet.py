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


class ImageNet(Dataset):

    def __init__(self, root, image_set, transform, num_classes=1000, lmdb_path=None, full_data=False, **kwargs):
        super().__init__()
        self.root = osp.join(root, 'imagenet-1k')
        self.image_set = image_set
        self.transform = transform
        self.lmdb_path = lmdb_path
        self.lmdb_txn = None
        self.full_data = full_data

        if num_classes == 1000:
            class_names = os.listdir(os.path.join(self.root, image_set))
            class_names = sorted(class_names)
        elif num_classes == 100:
            with open(os.path.join(self.root, 'imagenet100.txt'), 'r') as f:
                class_names = f.read().splitlines()
        else:
            assert False
        self.file_list = self.make_dataset(class_names)

    def make_dataset(self, class_names):
        instances = []
        for label, target_class in enumerate(class_names):
            target_dir = os.path.join(self.root, self.image_set, target_class)
            if not os.path.isdir(target_dir):
                raise ValueError(f"Target class {target_class} could not be found under path {target_dir}")
            if self.full_data:
                max_length_per_class = 1000000
            else:
                max_length_per_class = 30
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames)[:max_length_per_class]:
                    path = os.path.join(root, fname)
                    instances.append((path, label))

        return instances

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, lbl = self.file_list[idx]
        if self.lmdb_path is not None:
            if self.lmdb_txn is None:
                lmdb_dir = self.lmdb_path
                build_lmdb(lmdb_dir, [i[0] for i in self.file_list])
                env = lmdb.open(lmdb_dir, map_size=int(1e12), readonly=True, lock=False, readahead=True, meminit=False)
                self.lmdb_txn = env.begin(write=False)
                self.meta_info = pickle.load(open(os.path.join(lmdb_dir, 'meta_info.pkl'), "rb"))

            img_buff = self.lmdb_txn.get(img_path.encode('ascii'))
            C, H, W = [int(i) for i in self.meta_info[img_path].split('_')]
            img = np.frombuffer(img_buff, dtype=np.uint8)
            img = torch.from_numpy(img.copy())
            img = img.float().reshape(C, H, W)
            img = img.div(255)
        else:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img).astype(dtype=np.uint8)
            img = torch.from_numpy(img).float().permute(2,0,1)
            img = img.div(255)

        if self.transform:
            image = self.transform(img)

        return image, lbl
