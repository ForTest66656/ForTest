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

from .hrnet import HighResolutionNet, BasicBlock, Bottleneck
from .lddmm import *
from ..utils.lddmm_params import get_index


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class LDDMMHighResolutionNet(HighResolutionNet):
    def __init__(self, config, deform=True, stage=0, **kwargs):
        # super(LDDMMHighResolutionNet, self).__init__(config)

        self.config = config
        # self.return_momentum = False
        self.return_momentum = (config.TEST.DATASET != config.DATASET.DATASET) and (config.TEST.DATASET != 'COFW')
        self.is_train = not config.TEST.INFERENCE
        self.lddmm = deform
        self.points = config.MODEL.NUM_JOINTS if self.is_train else config.TEST.NUM_JOINTS
        self.stage = stage
        self.scale = config.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.train_fe = config.MODEL.FINETUNE_FE
        self.inplanes = 64
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Regression Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.regressor = nn.Linear(2048, config.MODEL.NUM_JOINTS*2)
        
        if config.DATASET.DATASET == '300W' or config.DATASET.DATASET == 'COFW':
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
            not self.return_momentum and (config.TEST.DATASET != 'COFW'):
            self.index = get_index(config.DATASET.DATASET, self.points)
            self.init_landmarks = self.init_landmarks[self.index]

        self.landmark2img = LandmarkImageLayer(patch_size=6)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward_function(self, x, init_pts=None):
        # h, w = x.size(2), x.size(3)
        if init_pts is None:
            init_pts = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)

        with torch.set_grad_enabled(self.train_fe):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.layer1(x)

            x_list = []
            for i in range(self.stage2_cfg['NUM_BRANCHES']):
                if self.transition1[i] is not None:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(x)
            y_list = self.stage2(x_list)

            x_list = []
            for i in range(self.stage3_cfg['NUM_BRANCHES']):
                if self.transition2[i] is not None:
                    x_list.append(self.transition2[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage3(x_list)

            x_list = []
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        # Regression Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                [2:]).view(y.size(0), -1)

        alpha = self.regressor(y)
        if self.lddmm and not self.return_momentum:
            return self.deform(alpha, init_pts)
        if self.return_momentum:
            return self.cross_deform(self.config, alpha.view(alpha.size(0), -1, 2))
        else:
            return alpha.view(alpha.size(0), -1, 2)

    def forward(self, x, init_pts=None):
        output = self.forward_function(x, init_pts)
        return output

    def cross_deform(self, config, momentum):
        batch_size = momentum.size(0)
        sigmaV2 = get_sigmaV2(config.DATASET.DATASET, config.MODEL.NUM_JOINTS)
        curve_deform = SingleCurveDeformLayer()

        if config.DATASET.DATASET == '300W' and config.TEST.DATASET == 'Helen':
            source_init_landmarks = np.load('data/init_landmark.npy')
            source_init_landmarks -= 56
            source_init_landmarks *= 1.25
            source_init_landmarks += 56
            target_init_landmarks1 = scipy.io.loadmat('data/300w/Helen_to_300w_baseShape.mat')['shape']
            target_init_landmarks2 = scipy.io.loadmat('data/300w/Helen_to_300w_finalShape_fineTuned.mat')['data_Helen']
            target_init_landmarks = np.concatenate((target_init_landmarks2[0:41], target_init_landmarks1[41:58], target_init_landmarks2[58:]), axis=0)
            target_init_landmarks -= 56
            target_init_landmarks *= 1.25
            target_init_landmarks += 56
            # s2t = [0, 1, 10, 9, -1, -1, 8, 7, 3, 4, 5, 6]
            s2t = [0, 1, 11, 10, 2, 3, 9, 8, 4, 5, 6, 7]
        elif config.DATASET.DATASET == 'Helen' and config.TEST.DATASET == '300W':
            source_init_landmarks = scipy.io.loadmat('data/300w/images/helen/Helen_meanShape_256_1_5x.mat')['Helen_meanShape_256_1_5x']
            source_init_landmarks *= (112 / 256)
            target_init_landmarks1 = scipy.io.loadmat('data/300w/images/helen/300w_to_Helen_baseShape.mat')['shape']
            target_init_landmarks2 = scipy.io.loadmat('data/300w/images/helen/300w_to_Helen_finalShape_fineTuned.mat')['data_300w']
            target_init_landmarks = np.concatenate((target_init_landmarks1[0:36], target_init_landmarks2[36:]), axis=0)
            target_init_landmarks *= (112 / 256)
            # s2t = [0, 1, -1, 8, 9, 10, 11, 7, 6, 3, 2]
            s2t = [0, 1, [4, 5], 8, 9, 10, 11, 7, 6, 3, 2]
        elif config.DATASET.DATASET == '300W' and config.TEST.DATASET == 'WFLW':
            source_init_landmarks = np.load('data/init_landmark.npy')
            source_init_landmarks -= 56
            source_init_landmarks *= 1.25
            source_init_landmarks += 56
            target_init_landmarks = scipy.io.loadmat('data/300w/WFLW_to_300w_finalShape_fineTuned.mat')['data_WFLW']
            target_init_landmarks1 = np.concatenate((target_init_landmarks[0:68], target_init_landmarks[69:77], target_init_landmarks[78:]), axis=0)
            target_init_landmarks2 = scipy.io.loadmat('data/300w/WFLW_to_300w_finalShape.mat')['shape_final']
            target_init_landmarks = np.concatenate((target_init_landmarks2[0:60], target_init_landmarks1[60:76], target_init_landmarks2[76:]), axis=0)
            target_init_landmarks -= 56
            target_init_landmarks *= 1.25
            target_init_landmarks += 56
            s2t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif config.DATASET.DATASET == 'WFLW' and config.TEST.DATASET == '300W':
            source_init_landmarks = np.load('data/wflw/init_landmark.npy')
            target_init_landmarks = scipy.io.loadmat('data/wflw/300w_to_WFLW_finalShape_fineTuned.mat')['data_300w']
            s2t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        
        source_curve2landmark = get_curve2landmark(config.DATASET.DATASET, config.MODEL.NUM_JOINTS)
        target_curve2landmark = get_curve2landmark(config.TEST.DATASET, config.TEST.NUM_JOINTS)

        source_init_landmarks = torch.Tensor(source_init_landmarks).cuda().view(1, -1, 2)
        source_init_landmarks = torch.cat([source_init_landmarks]*batch_size, dim=0)
        source_init_landmarks = source_init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 112
        target_init_landmarks = torch.Tensor(target_init_landmarks).cuda().view(1, -1, 2)
        target_init_landmarks = torch.cat([target_init_landmarks]*batch_size, dim=0)
        target_init_landmarks = target_init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 112
        results = torch.ones([batch_size, config.TEST.NUM_JOINTS, 2]).cuda()

        for k, v in source_curve2landmark.items():
            if isinstance(s2t[k], list):
                for i in s2t[k]:
                    target_curve = target_curve2landmark[i]
                    results[:, target_curve] = curve_deform(momentum[:, v], source_init_landmarks[:, v], target_init_landmarks[:, target_curve], sigmaV2[v][0])
            elif s2t[k] != -1:
                target_curve = target_curve2landmark[s2t[k]]
                results[:, target_curve] = curve_deform(momentum[:, v], source_init_landmarks[:, v], target_init_landmarks[:, target_curve], sigmaV2[v][0])

        return results