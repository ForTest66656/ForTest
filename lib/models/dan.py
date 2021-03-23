from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg11_bn
from .lddmm import *
from ..utils.lddmm_params import get_index

logger = logging.getLogger(__name__)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias, padding=1)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class DAN(nn.Module):
    def __init__(self, config, deform=True, **kwargs):
        super(DAN, self).__init__()

        self.config = config
        # self.return_momentum = False
        self.return_momentum = (config.TEST.DATASET != config.DATASET.DATASET)
        self.is_train = not config.TEST.INFERENCE
        self.lddmm = deform
        self.points = config.MODEL.NUM_JOINTS if self.is_train else config.TEST.NUM_JOINTS
        self.scale = config.DATASET.BOUNDINGBOX_SCALE_FACTOR

        # deep alignment network
        self.stage1 = nn.Sequential(
            BasicBlock(nn.Conv2d, 3, 64, 3),
            BasicBlock(nn.Conv2d, 64, 64, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 64, 128, 3),
            BasicBlock(nn.Conv2d, 128, 128, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 128, 256, 3),
            BasicBlock(nn.Conv2d, 256, 256, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 256, 512, 3),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(8*8*512, 2048),
            nn.ReLU(True)
        )
        self.head1 = nn.Linear(2048, config.MODEL.NUM_JOINTS*2)
        self.project = nn.Sequential(
            nn.Linear(2048, 64*64),
            nn.ReLU(True)
        )
        self.stage2 = nn.Sequential(
            BasicBlock(nn.Conv2d, 5, 64, 3),
            BasicBlock(nn.Conv2d, 64, 64, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 64, 128, 3),
            BasicBlock(nn.Conv2d, 128, 128, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 128, 256, 3),
            BasicBlock(nn.Conv2d, 256, 256, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 256, 512, 3),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            nn.MaxPool2d((2, 2)),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            BasicBlock(nn.Conv2d, 512, 512, 3),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(8*8*512, 2048),
            nn.ReLU(True)
        )
        self.head2 = nn.Linear(2048, config.MODEL.NUM_JOINTS*2)
        self.img_affine = AffineTransformLayer()
        self.landmark_affine = LandmarkTransformLayer()
        self.estimate = TransformParamsLayer()
        self.landmark2img = LandmarkImageLayer(patch_size=6)
        
        # deformation configuration
        if config.DATASET.DATASET == '300W':
            self.init_landmarks = np.load('data/init_landmark.npy')
            if self.points == 131:
                self.init_landmarks = scipy.io.loadmat('data/300w/upsample_131.mat')['upsample_131']
            self.init_landmarks -= 56
            self.init_landmarks *= 1.25
            self.init_landmarks += 56
        elif config.DATASET.DATASET == 'WFLW':
            self.init_landmarks = np.load('data/wflw/init_landmark.npy')
        elif config.DATASET.DATASET == 'Helen':
            self.init_landmarks = scipy.io.loadmat('data/300w/images/helen/Helen_meanShape_256_1_5x.mat')['Helen_meanShape_256_1_5x']
            self.init_landmarks *= (112 / 256)
        
        logger.info('=> loading initial landmarks...')
        self.init_landmarks = torch.Tensor(self.init_landmarks).cuda()
        self.init_landmarks = self.init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 112
         
        if self.is_train:
            self.deform = LandmarkDeformLayer(config, n_landmark=config.MODEL.NUM_JOINTS)
        elif self.return_momentum:
            self.deform = None
        else:
            self.deform = LandmarkDeformLayer(config, n_landmark=config.TEST.NUM_JOINTS)
                    
        if ((config.DATASET.DATASET == '300W' and self.points < 131) or \
           (config.DATASET.DATASET == 'WFLW' and self.points < 98) or \
           (config.DATASET.DATASET == 'Helen' and self.points < 194)) and \
            not self.return_momentum:
            self.index = get_index(config.DATASET.DATASET, self.points)
            self.init_landmarks = self.init_landmarks[self.index]

    def forward(self, x, init_pts=None):
        if init_pts is None:
            init_pts = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)

        # stage 1
        fe1 = self.stage1(x)
        pred1 = self.head1(fe1)
        if self.lddmm:
            pred1 = self.deform(pred1, init_pts)
        else:
            pred1 = init_pts.view(x.size(0), -1) + pred1
        spatial_fe1 = self.project(fe1)
        params = self.estimate(pred1, init_pts)
        deformed_x = self.img_affine(x, params)
        affined_preds = self.landmark_affine(pred1, params)
        # 1/4 heatmap size
        x_downsample = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=False)
        heatmap = self.landmark2img(x_downsample, affined_preds)
        heatmap = F.interpolate(heatmap, size=[256, 256], mode='bilinear', align_corners=False)
        spatial_fe1 = F.interpolate(spatial_fe1.view(-1, 1, 64, 64), size=[256, 256], mode='bilinear', align_corners=False)

        # stage 2
        x2 = torch.cat([deformed_x, spatial_fe1, heatmap], dim=1)
        fe2 = self.stage2(x2)
        pred2 = self.head2(fe2)
        if self.lddmm:
            pred2 = self.deform(pred2, affined_preds.view(x.size(0), -1, 2))
        else:
            pred2 = affined_preds + pred2
        pred2 = self.landmark_affine(pred2, params, inverse=True)

        return [pred1.view(pred1.size(0), -1, 2), pred2.view(pred2.size(0), -1, 2)]