# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_zone_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import numpy as np

@DATASETS.register_module()
class VOCDataset(XMLDataset):
    '''
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    '''
    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

    #CLASSES = ('nomask','mask')	#face mask dataset
    CLASSES = ('apple','orange','pear','watermelon','durian','lemon','grape','pineapple','pitaya','muskmelon','melon')	#fruit dataset
    #CLASSES = ('head','helmet')	#helmet dataset

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        data = self.data_infos

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            total_ap = 0
            total_ar = 0
            # total_map=0
            map = []
            mar = []
            aps = np.zeros([len(self.CLASSES)])
            for iii in range(10):
                mean_ap, mean_ar, cls_aps, _ = eval_map(
                    results,
                    annotations,
                    data,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)

                aps = cls_aps + aps
                iou_thr = iou_thr + 0.05
                total_ap = mean_ap + total_ap
                total_ar = mean_ar + total_ar
                if iii == 0:
                    eval_results['mAP'] = mean_ap
                    eval_results['mAR'] = mean_ar
                map.append('AP' + str(int(50 + 5 * iii)) + ': ' +
                           str(format(mean_ap, '.3f')))
                mar.append('AR' + str(int(50 + 5 * iii)) + ': ' +
                           str(format(mean_ar, '.3f')))
            print('AP: ', total_ap / 10)
            print('AR: ', total_ar / 10)
            map.append('AP' +  ': ' + str(format(total_ap/10, '.3f')))
            mar.append('AR' +  ': ' + str(format(total_ar/10, '.3f')))
            print(map)
            print(mar)
            print(np.around(aps/10, 3))
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def zone_evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 ri=0.0, rj=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        data = self.data_infos
        print('------------------------------')
        print('ri=', ri, 'rj=', rj)
        print(metric)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}

        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            total_ap = 0
            total_ar = 0
            map = np.zeros([11])
            mar = np.zeros([11])
            aps = np.zeros([len(self.CLASSES)])
            for iii in range(10):
                mean_ap, mean_ar, cls_aps, _ = eval_zone_map(
                    results,
                    annotations,
                    data,
                    ri,
                    rj,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)

                aps = cls_aps + aps
                # total_map=mean_ap+total_map
                iou_thr = iou_thr + 0.05
                total_ap = mean_ap + total_ap
                total_ar = mean_ar + total_ar
                if iii == 0:
                    eval_results['mAP'] = mean_ap
                    eval_results['mAR'] = mean_ar
                map[iii]=format(mean_ap, '.3f')
                mar[iii]=format(mean_ar, '.3f')
            print('AP: ', total_ap / 10)
            print('AR: ', total_ar / 10)
            map[10]=format(total_ap/10, '.3f')
            mar[10]=format(total_ar/10, '.3f')
            print(map)
            print(mar)
            aps = np.around(aps/10, 3)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results, map*100, mar*100, aps*100

