import math
import random

import torch
import torch.utils.data as data
import numpy as np

from hdf5storage import loadmat
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
from lib.utils.lddmm_params import get_index


class COFW(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.mat_file = cfg.DATASET.TRAINSET
        else:
            self.mat_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.dataset_name = 'COFW'
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.points = cfg.MODEL.NUM_JOINTS if is_train else cfg.TEST.NUM_JOINTS
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.bounding_box_scale_factor = cfg.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.index = get_index('300W', self.points)

        # load annotations
        self.mat = loadmat(self.mat_file)
        if is_train:
            self.images = self.mat['IsTr']
            self.pts = self.mat['phisTr']
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        # pts = self.pts[idx][0:58].reshape(2, -1).transpose()
        pts = loadmat('./data/cofw/test_annotations/{}_points.mat'.format(idx+1))['Points']

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        box_size = np.sqrt((xmax - xmin)*(ymax - ymin))

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        center = torch.Tensor([center_w, center_h])

        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='COFW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        if self.label_type == 'Gaussian':
            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                if self.label_type == 'Gaussian':                               
                    target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                                label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        if self.label_type == 'Gaussian':
            target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        origin_pts = torch.Tensor(pts)

        # weak-supervised
        if self.points < 68:
            tpts = tpts[self.index]
            pts = pts[self.index]
            if self.label_type == 'Gaussian':
                target = target[self.index]

        meta = {'index': idx, 'center': center, 'scale': scale, 'dataset_name': self.dataset_name,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'origin_pts': origin_pts, 'box_size': box_size}

        if self.label_type == 'Gaussian':
            return img, target, meta
        else:
            return img, tpts, meta


if __name__ == '__main__':

    pass
