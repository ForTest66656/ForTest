from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import HighResolutionNet
from .lddmm_hrnet import LDDMMHighResolutionNet
from .hourglass import HourglassNet
from .lddmm_hourglass import LDDMM_Hourglass
from .dan import DAN


def get_face_alignment_net(config, **kwargs):
    if config.MODEL['NAME'] == 'hrnet':
        model = HighResolutionNet(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'lddmm_hrnet':
        model = LDDMMHighResolutionNet(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'coord_hrnet':
        model = LDDMMHighResolutionNet(config, deform=False, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'hourglass':
        model = HourglassNet(config, **kwargs)
    elif config.MODEL['NAME'] == 'lddmm_hourglass':
        model = LDDMM_Hourglass(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'coord_hourglass':
        model = LDDMM_Hourglass(config, deform=False, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'dan':
        model = DAN(config, deform=False, **kwargs)
    elif config.MODEL['NAME'] == 'lddmm_dan':
        model = DAN(config, **kwargs)
    else:
        raise NotImplementedError('{} is not available'.format(config.model['NAME']))

    return model

__all__ = ['HighResolutionNet', 'LDDMMHighResolutionNet', 'get_face_alignment_net']
