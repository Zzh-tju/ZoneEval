## Implementation of Zone Evaluation

Let’s see what this looks like in practice.

### Overall process

We keep the main body of mmdetection unchanged. All the functions for zone evaluation are copied from the original ones.
For examples, 

`get_bboxes()` $\rightarrow$ `get_zone_bboxes()`, 

`_get_bboxes_single()` $\rightarrow$ `_get_zone_bboxes_single()`,

`_bbox_post_process()` $\rightarrow$ `_zone_bbox_post_process()`.

Please refer to [base_dense_head.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/dense_heads/base_dense_head.py).
In the following, we take one-stage dense detector as example. (For multi-stage detectors, it is similar.
Please take a look at [two_stage.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/detectors/two_stage.py#L187) and [standard_roi_head.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/roi_heads/standard_roi_head.py#L269).)

In [_zone_bbox_post_process](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/dense_heads/base_dense_head.py#L507),
we define zones as a series of annular regions. You can define the zone shape to be whatever you want.

Then, in [single_stage.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/detectors/single_stage.py#L115), it recieves 5 zone detections.

In [multi_gpu_test](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/apis/test.py#L81),
it packs up the 5 zone detections.

In [test.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/tools/test.py#L273), the function `zone_evaluate()` evaluates the zone detections and return the zone metrics.

For VOC, the ignored ground-truth boxes are defined in [mean_ap.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/core/evaluation/mean_ap.py#L265). For MS COCO, it lies in [cocoeval.py](https://github.com/Zzh-tju/ZoneEval/blob/main/pycocotools/pycocotools/cocoeval.py#L596).

### Detailed process

Add `ri, rj, img_w, img_h` in the `_zone_bbox_post_process()` function of `/mmdet/models/dense_heads/base_dense_head.py`.
```python
from mmdet.core import bbox_xyxy_to_cxcywh
    def _zone_bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           ri,       # the range of zones
                           rj,
                           img_w,       # the width and height of image
                           img_h,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
    ......
    
         if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]

            # After NMS, select the detections in the zone
            xywh_det = bbox_xyxy_to_cxcywh(det_bboxes[:,:4])
            ind0 = (xywh_det[:,0] <= rj*img_w) | ( xywh_det[:,0] >= (1-rj)*img_w) | ( xywh_det[:,1] <= rj*img_h) | ( xywh_det[:,1] >= (1-rj)*img_h)
            ind1 = (xywh_det[:,0] > ri*img_w) & ( xywh_det[:,0] < (1-ri)*img_w) & ( xywh_det[:,1] > ri*img_h) & ( xywh_det[:,1] < (1-ri)*img_h)
            ind2 = ind0 & ind1
            det_bboxes=det_bboxes[ind2]
            det_labels=det_labels[ind2]

            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels
```
Note that the above modification is just an example, and you should modify several relevant files to keep the inference normal, all because the post-process consists of several functions, e.g., `_get_zone_bboxes_single()` and `get_zone_bboxes()` in `/mmdet/models/dense_heads/base_dense_head.py`.

Then we modify `mmdet/models/detectors/single_stage.py`,

```python
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

    ......

        if self.test_cfg.zone_measure==False:
            return bbox_results, None, None, None, None, None
        else:
            bbox_list0_1 = self.bbox_head.get_zone_bboxes(
                *outs, img_metas=img_metas, ri=0.0, rj=0.1, rescale=rescale)
            bbox_list1_2 = self.bbox_head.get_zone_bboxes(
                *outs, img_metas=img_metas, ri=0.1, rj=0.2, rescale=rescale)
            bbox_list2_3 = self.bbox_head.get_zone_bboxes(
                *outs, img_metas=img_metas, ri=0.2, rj=0.3, rescale=rescale)
            bbox_list3_4 = self.bbox_head.get_zone_bboxes(
                *outs, img_metas=img_metas, ri=0.3, rj=0.4, rescale=rescale)
            bbox_list4_5 = self.bbox_head.get_zone_bboxes(
                *outs, img_metas=img_metas, ri=0.4, rj=0.5, rescale=rescale)
            # skip post-processing when exporting to ONNX
            if torch.onnx.is_in_onnx_export():
                return bbox_list, bbox_list0_1, bbox_list1_2, bbox_list2_3, bbox_list3_4, bbox_list4_5

            bbox_results0_1 = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list0_1]

            bbox_results1_2 = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list1_2]

            bbox_results2_3 = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list2_3]

            bbox_results3_4 = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list3_4]

            bbox_results4_5 = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list4_5]

            return bbox_results, bbox_results0_1, bbox_results1_2, bbox_results2_3, bbox_results3_4, bbox_results4_5
```
and `mmdet/apis/test.py`,

```python
def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

......
    # results for the 5 zones
    results0_1 = []
    results1_2 = []
    results2_3 = []
    results3_4 = []
    results4_5 = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, result0_1, result1_2, result2_3, result3_4, result4_5 = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

            results.extend(result)
            if result0_1:
                if isinstance(result0_1[0], tuple):
                    result0_1 = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result0_1]

                if isinstance(result1_2[0], tuple):
                    result1_2 = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result1_2]

                if isinstance(result2_3[0], tuple):
                    result2_3 = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result2_3]

                if isinstance(result3_4[0], tuple):
                    result3_4 = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result3_4]

                if isinstance(result4_5[0], tuple):
                    result4_5 = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result4_5]

                results0_1.extend(result0_1)
                results1_2.extend(result1_2)
                results2_3.extend(result2_3)
                results3_4.extend(result3_4)
                results4_5.extend(result4_5)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        if result0_1:
            results0_1 = collect_results_gpu(results0_1, len(dataset))
            results1_2 = collect_results_gpu(results1_2, len(dataset))
            results2_3 = collect_results_gpu(results2_3, len(dataset))
            results3_4 = collect_results_gpu(results3_4, len(dataset))
            results4_5 = collect_results_gpu(results4_5, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        if result0_1:
            results0_1 = collect_results_cpu(results0_1, len(dataset), tmpdir)
            results1_2 = collect_results_cpu(results1_2, len(dataset), tmpdir)
            results2_3 = collect_results_cpu(results2_3, len(dataset), tmpdir)
            results3_4 = collect_results_cpu(results3_4, len(dataset), tmpdir)
            results4_5 = collect_results_cpu(results4_5, len(dataset), tmpdir)
    return results, results0_1, results1_2, results2_3, results3_4, results4_5
```

Also, in `tools/test.py`,

```python
      ......
        outputs, outputs0_1, outputs1_2, outputs2_3, outputs3_4, outputs4_5 = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))
        ......
                if cfg.evaluation.metric == 'bbox':
                    ZP = np.zeros([5,7])
                    ZP_class = np.zeros([5,num_class])
                    eval_results, metric, ap_class, _= dataset.zone_evaluate(outputs, **eval_kwargs, ri=0.0, rj=0.5)
                    eval_results0_1, metric0_1, ap_class0_1, _ = dataset.zone_evaluate(outputs0_1, **eval_kwargs, ri=0.0, rj=0.1)
                    eval_results1_2, metric1_2, ap_class1_2, _ = dataset.zone_evaluate(outputs1_2, **eval_kwargs, ri=0.1, rj=0.2)
                    eval_results2_3, metric2_3, ap_class2_3, _ = dataset.zone_evaluate(outputs2_3, **eval_kwargs, ri=0.2, rj=0.3)
                    eval_results3_4, metric3_4, ap_class3_4, _ = dataset.zone_evaluate(outputs3_4, **eval_kwargs, ri=0.3, rj=0.4)
                    eval_results4_5, metric4_5, ap_class4_5, _ = dataset.zone_evaluate(outputs4_5, **eval_kwargs, ri=0.4, rj=0.5)
      ......
```

In addition to getting the detections in the zone, we must select the ground-truth boxes in the zone either.
The relevant modification lies in `/mmdet/datasets/coco.py` (take COCO as example)

```python 
from .api_wrappers import COCO, COCOeval, COCO_zone_eval

# Add zone_evaluate() function and 

    def evaluate_zone_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None,
                          ri=0.0,
                          rj=0.5):

        ......
            cocoEval = COCO_zone_eval(coco_gt, coco_det, iou_type, self.coco.imgs, ri, rj)
```

In [cocoeval.py](https://github.com/Zzh-tju/ZoneEval/blob/main/pycocotools/pycocotools/cocoeval.py), we copy the class `COCOeval` to a new one `COCO_zone_eval`.

Then,
```python
class COCO_zone_eval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', imgs=None, ri=0.0, rj=0.5):
        self.imgs = imgs
        self.ri = ri 
        self.rj= rj

    def _prepare(self):
        ......
        # set ignore flag
        for gt in gts:
            img_w = self.imgs[gt['image_id']]['width']
            img_h = self.imgs[gt['image_id']]['height']
            center_x = gt['bbox'][0] + 0.5 * gt['bbox'][2]
            center_y = gt['bbox'][1] + 0.5 * gt['bbox'][3]
            ind0 = (center_x <= self.ri*img_w) | ( center_x >= (1-self.ri)*img_w) | ( center_y <= self.ri*img_h) | ( center_y >= (1-self.ri)*img_h)
            ind1 = (center_x > self.rj*img_w) & ( center_x < (1-self.rj)*img_w) & ( center_y > self.rj*img_h) & ( center_y < (1-self.rj)*img_h)
            ind2 = ind0 | ind1
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if ind2 == True:
                gt['ignore'] = 1
```
where we set all the ground-truth boxes whose centers are outside the zone to be 'ignored'.

Of course, we pack up `pycocotools` in our repository.

Once we get ZP of 5 zones, we calculate the variance of ZP to measure the degree of discreteness for zone performance.

```python
ZP_var = np.var(ZP, axis=0)
```

Finally, we calculate [Spatial equilibrium Precision](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/tools/test.py#L393) (SP) by computing the area weighted sum of ZP.

```python
SP = 0.36 * ZP[0,:] + 0.28 * ZP[1,:] + 0.2 * ZP[2,:] + 0.12 * ZP[3,:] + 0.04 * ZP[4,:]
```

We expect the detector to perform uniformly and well in all zones.

#### OK, let's have fun with zone evaluation.

## Implementation of SELA

### SELA (frequency-based approach)

1. `configs/sela/gfl_sela_r18_fpn_1x_voc.py`

```python
model = dict(
    train_cfg = dict(
        assigner=dict(type='ATSSAssigner', topk=9, gamma=0.2),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

# where gamma=0.2 is particularly chosen for VOC, while 0.1 by default for all the other datasets.
# Of course, you can search a better gamma for different application scenarios.
```

2. `mmdet/models/dense_heads/gfl_sela_head.py`

```python
    def _get_target_single(......):
        # get the image width and height, that must match the bbox coordinates.
        img_w = img_meta['img_shape'][1]
        img_h = img_meta['img_shape'][0]
        ......
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels, img_w=img_w, img_h=img_h)
        ......
```
3. `mmdet/core/bbox/assigners/atss_assigner.py`

```python
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None,
               img_w=0,    # add the two variables
               img_h=0):
        ......
        if self.gamma is not None:
            spatial_weight = 2*torch.max(torch.abs(gt_cx - 0.5*img_w)/img_w, torch.abs(gt_cy - 0.5*img_h)/img_h)
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :] - self.gamma * spatial_weight    # SELA
        else:
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]    # ATSS
        ......
```

### SELA (cost-sensitive learning)

1. `configs/sela/gfl_sela_cost_sensitive_learning_r18_fpn_1x_voc.py`

```python
model = dict(
    bbox_head=dict(gamma=0.1))
```

2. `mmdet/models/dense_heads/gfl_sela_head.py`

```python
    def _get_target_single(......):
        ......
        spatial_weights = anchors.new_ones(num_valid_anchors, dtype=torch.float)
        xywh_anchors = bbox_xyxy_to_cxcywh(anchors)
        if self.gamma is not None:
            spatial_weights = 2*torch.max(torch.abs(xywh_anchors[:,0] - 0.5*img_w)/img_w, torch.abs(xywh_anchors[:,1] - 0.5*img_h)/img_h)
```

This variable `spatial_weights` will be processed by the same process as `label_weights`.

```python
    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, spatial_weights,
                    bbox_targets, stride, num_total_samples):
        ...... 
        spatial_weights = spatial_weights.reshape(-1)

        if len(pos_inds) > 0:
            # SELA on bbox regression loss
            if self.gamma is not None:
                weight_targets = weight_targets * (spatial_weights[pos_inds] * self.gamma + 1)
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

        # SELA on classification loss
        if self.gamma is not None:
            label_weights = label_weights[pos_inds] * (spatial_weights[pos_inds] * self.gamma + 1)
        loss_cls =  self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)
```
