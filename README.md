<img src="flyleaf.png"/>

# Why does spatial equilibrium matter?

### The detector cannot perform uniformly across the zones.
<img src="detection-quality.png"/>

### The detection performance is correlated with the object distribution.

When the object distribution satisfies the centralized photographer’s bias, the detector will favor more to the central zone, while losing the performance in most areas outside.

### This is not good for robust detection application.

If you have a fire dataset like this, the detector will be good at detecting fire in the central zone of the image. But for the zone near to the image border, uh huh, hope you are safe.

<img src="fire-data.png" width="600"/>
<img src="fire.png" width="600"/>

## Zone Evaluation

### Zone Precision

Let’s start by the definition of evaluation zones. We define a rectangle region $R_i=\text{Rectangle}(p,q)=\text{Rectangle}((r_iW,r_iH),((1-r_i)W,(1-r_i)H))$ like this, 
<div align="center"><img src="rectangle.png" width="300"/></div>

where $i\in$ { $0,1,\cdots,n$ }, $n$ is the number of zones.

Then, the evaluation zones are disigned to be a series of annular zone $z_i^j=R_i\setminus R_j$, $i\textless j$.
We denote the range of the annular zone $z_i^j$ as $(r_i,r_j)$ for brevity.

<div align="center"><img src="zone-range.gif" width="300"/></div>

We measure the detection performance for a specific zone $z_i^j$ by only considering the ground-truth objects and the detections whose centers lie in the zone $z_i^j$.
Then, for an arbitrary evaluation metric $m$, for instance Average Precision (AP), the evaluation process stays the same to the conventional ways, yielding Zone Precision (ZP), denoted by ZP@ $z_i^j$. Consider the default setting $n=5$, the evaluation zones look like this,

<div align="center"><img src="eval-zone.png" width="300"/></div>

#### Implementation

Let’s see what this looks like in practice.

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

Then, in `tools/test.py`,

```python
      ......
        outputs, outputs0_1, outputs1_2, outputs2_3, outputs3_4, outputs4_5 = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))
                
                if cfg.evaluation.metric == 'bbox':
                    eval_results, metric, _, _= dataset.zone_evaluate(outputs, **eval_kwargs, ri=0.0, rj=0.5)
                    eval_results0_1, metric0_1, _, _ = dataset.zone_evaluate(outputs0_1, **eval_kwargs, ri=0.0, rj=0.1)
                    eval_results1_2, metric1_2, _, _ = dataset.zone_evaluate(outputs1_2, **eval_kwargs, ri=0.1, rj=0.2)
                    eval_results2_3, metric2_3, _, _ = dataset.zone_evaluate(outputs2_3, **eval_kwargs, ri=0.2, rj=0.3)
                    eval_results3_4, metric3_4, _, _ = dataset.zone_evaluate(outputs3_4, **eval_kwargs, ri=0.3, rj=0.4)
                    eval_results4_5, metric4_5, _, _ = dataset.zone_evaluate(outputs4_5, **eval_kwargs, ri=0.4, rj=0.5)
      ......
```

The same process for selecting the ground-truth boxes in the zone lies in `/mmdet/datasets/coco.py`


### Spatial Equilibrium Precision

Now that we have 5 ZPs, and they indeed provide more information about the detector's performance. We further present a **S**patial equilibrium **P**recision (SP), and we use this single value to characterize the detection performance for convenient usage.

<div align="center"> $\mathrm{SP}=\sum\limits_{i=0}^{n-1}\mathrm{Area}(z_i^{i+1})\mathrm{ZP}\text{@}z_i^{i+1}$ </div>

where $\mathrm{Area}(z)$ calculates the area of the zone $z_i^j$ in the normalized image space (square image with unit area 1). In general, SP is a weighted sum of the
 5 ZPs, that is,
 
<div align="center"> $\mathrm{SP}=0.36\mathrm{ZP}\text{@}z_0^1+0.28\mathrm{ZP}\text{@}z_1^2+0.20\mathrm{ZP}\text{@}z_2^3+0.12\mathrm{ZP}\text{@}z_3^4+0.04\mathrm{ZP}\text{@}z_4^5$ </div>

Our SP is based on the assumption similar to the traditional AP, i.e., the detector performs uniformly in the zone.
The difference is, our SP applies this assumption to a series of smaller zones, rather than the full map for traditional AP.
One can see that when $n=1$, our SP is identical to traditional AP as the term $\mathrm{Area}(z_i^j)=1$, which means that the detectors are assumed to perform uniformly in the whole image zone.
As $n$ increases, the requirements for spatial equilibrium become stricter and stricter. And a large $n>5$ is also acceptable if a more rigorous spatial equilibrium is required.

