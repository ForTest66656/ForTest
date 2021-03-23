from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW
from .helen import Helen

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'Helen', 'get_dataset', 'get_testset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'Helen':
        return Helen
    else:
        raise NotImplemented()

def get_testset(config):

    if config.TEST.DATASET == 'AFLW':
        return AFLW
    elif config.TEST.DATASET == 'COFW':
        return COFW
    elif config.TEST.DATASET == '300W':
        return Face300W
    elif config.TEST.DATASET == 'WFLW':
        return WFLW
    elif config.TEST.DATASET == 'Helen':
        return Helen
    else:
        raise NotImplemented()

