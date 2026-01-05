import os
import os.path as osp
import lmdb
import pickle
import numpy as np
import torch
import torch.multiprocessing
from random import shuffle

from PIL import Image
from torch.utils.data import Dataset

from .utils import build_lmdb
from .cityscapes import RandomCrop


class COCO(Dataset):
    def __init__(self, root, image_set, transform,
                 target_transform, random_crop_res, coarse_labels, exclude_things, exclude_stuff=False, subset=None, lmdb_path=None):
        super(COCO, self).__init__()
        self.split = image_set
        self.root = osp.join(root, "coco")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things
        self.exclude_stuff = exclude_stuff
        self.unlabel = False
        self.lmdb_path = lmdb_path
        self.lmdb_txn = None
        self.debug = random_crop_res

        self.random_crop = RandomCrop(random_crop_res, (0.64, 1))

        if self.subset is None:
            self.unlabel = True
            self.image_list = None
        elif self.subset == 0:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        if self.image_list is not None:
            self.image_files = []
            self.label_files = []
            for split_dir in split_dirs[self.split]:
                with open(osp.join(self.root, "curated", split_dir, self.image_list), "r") as f:
                    img_ids = [fn.rstrip() for fn in f.readlines()]
                    for img_id in img_ids:
                        self.image_files.append(osp.join(self.root, "images", split_dir, img_id + ".jpg"))
                        self.label_files.append(osp.join(self.root, "annotations", split_dir, img_id + ".png"))
        else:
            self.image_files = []
            self.label_files = None
            for split_dir in split_dirs[self.split]:
                img_ids = os.listdir(osp.join(self.root, "images", split_dir))
                img_ids = sorted(img_ids)
                for img_id in img_ids:
                    self.image_files.append(osp.join(self.root, "images", split_dir, img_id))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
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
            if self.label_files is not None:
                label_path = self.label_files[index]
                label = Image.open(label_path)
                img, label = self.random_crop(img, label)

            img = np.array(img).astype(dtype=np.uint8)
            img = torch.from_numpy(img).float().permute(2,0,1)
            img = img.div(255)

        if self.label_files is None:
            img = self.transform(img)
            return img,

        img = self.transform(img)
        label = self.label_transform(label).squeeze(0)
        # label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == 255] = 255
        # coarse_label = label

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            elif self.exclude_stuff:
                coarse_label[coarse_label >= self.first_stuff_index] = 255
                return img, coarse_label, coarse_label >=0
            else:
                return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)
