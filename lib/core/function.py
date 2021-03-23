from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import imageio
import cv2
import numpy as np

from .evaluation import compute_curve_dist, compute_perpendicular_dist, decode_preds, decode_duplicate, compute_nme, AUCError, get_index
from ..utils.transforms import transform_preds

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_collect = []
    nme_count = 0
    nme_batch_sum = 0
    curve_dist_batch_sum = 0
    curve_dist5_batch_sum = [0, 0, 0, 0, 0]
    pcurve_dist_batch_sum = 0
    pcurve_dist5_batch_sum = [0, 0, 0, 0, 0]
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            inp = inp.cuda()
            data_time.update(time.time() - end)
            output = model(inp)
            if isinstance(output, list):
                output = output[1]
                
            score_map = output.data.cpu()

            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)
            preds = decode_duplicate(preds, config)

            # NME
            nme_temp = compute_nme(preds, meta, config)
            nme_collect.extend(list(nme_temp))
            curve_dist_temp, curve_dist5_temp = compute_curve_dist(preds, meta)
            # pcurve_dist_temp, pcurve_dist5_temp = compute_perpendicular_dist(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            curve_dist_batch_sum += np.sum(curve_dist_temp)
            # pcurve_dist_batch_sum += np.sum(pcurve_dist_temp)
            for j in range(len(curve_dist5_batch_sum)):
                curve_dist5_batch_sum[j] += np.sum(curve_dist5_temp[j])
                # pcurve_dist5_batch_sum[j] += np.sum(pcurve_dist5_temp[j])
            nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]
            if config.TEST.SAVE_FIG:
                index = get_index(config.TEST.DATASET, config.MODEL.NUM_JOINTS)
                score_map = decode_duplicate(output.data.cpu(), config)
                # score_map = target
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                imgs = inp.cpu().detach().permute(0, 2, 3, 1).numpy()
                for idx in range(imgs.shape[0]):
                    img = np.array((imgs[idx].copy() * std + mean)*255).astype(np.float32)
                    tpts = (score_map[idx].numpy() * 256 / 112).astype(np.uint8)    
                    for k in range(tpts.shape[0]):
                        if k in index:
                            cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
                        else:
                            cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [255, 0, 0], -1)
                    imageio.imwrite('weak_300w_50/{}.jpg'.format(i*16+idx), img.astype(np.uint8))
                    print('weak_300w_50/{}.jpg'.format(i*16+idx))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    curve_dist = curve_dist_batch_sum / nme_count
    curve_dist5 = np.array(curve_dist5_batch_sum) / nme_count
    pcurve_dist = pcurve_dist_batch_sum / nme_count
    pcurve_dist5 = np.array(pcurve_dist5_batch_sum) / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    msg2 = 'Curve error:{:.5f} [Edge]:{:.5f} [Eyebrow]:{:.5f} [Nose]:{:.5f} [Eye]:{:.5f} [Mouth]:{:.5f}' \
            .format(curve_dist, curve_dist5[0], curve_dist5[1], curve_dist5[2], curve_dist5[3], curve_dist5[4])
    # msg3 = 'P-Curve error:{:.5f} [Edge]:{:.5f} [Eyebrow]:{:.5f} [Nose]:{:.5f} [Eye]:{:.5f} [Mouth]:{:.5f}' \
    #         .format(pcurve_dist, pcurve_dist5[0], pcurve_dist5[1], pcurve_dist5[2], pcurve_dist5[3], pcurve_dist5[4])
    logger.info(msg)
    logger.info(msg2)
    # logger.info(msg3)
    AUCError(nme_collect, 0.1, logger)

    return nme, predictions
