
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import cv2
import lmdb
import pickle
import fcntl

from PIL import Image
from tqdm import tqdm


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


def get_colormap(dataset_name):
    if dataset_name.startswith("cityscapes"):
        label_cmap = create_cityscapes_colormap()
    else:
        label_cmap = create_pascal_label_colormap()

    return label_cmap


def build_lmdb(save_path, metas, commit_interval=1000):
    with open('lock', 'w') as f:
        if not save_path.endswith('.lmdb'):
            raise ValueError("lmdb_save_path must end with 'lmdb'.")

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        if os.path.exists(save_path):
            # print('Folder [{:s}] already exists.'.format(save_path))
            return

        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))

        data_size_per_img = cv2.imread(metas[0], cv2.IMREAD_UNCHANGED).nbytes
        data_size = data_size_per_img * len(metas)
        env = lmdb.open(save_path, map_size=data_size * 10)
        txn = env.begin(write=True)

        shape = dict()

        print('Building lmdb...')
        for i in tqdm(range(len(metas))):
            image_filename = metas[i]
            img = Image.open(image_filename).convert("RGB")
            img = np.array(img).astype(dtype=np.uint8)
            img = img.transpose(2,0,1)
            # img = torch.from_numpy(img).permute(2,0,1)
            assert img is not None and len(img.shape) == 3 and img.shape[0] == 3

            txn.put(image_filename.encode('ascii'), img.copy(order='C'))
            shape[image_filename] = '{:d}_{:d}_{:d}'.format(img.shape[0], img.shape[1], img.shape[2])

            if i % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        pickle.dump(shape, open(osp.join(save_path, 'meta_info.pkl'), "wb"))

        txn.commit()
        env.close()
        print('Finish writing lmdb.')
