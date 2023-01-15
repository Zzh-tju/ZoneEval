### This is repo is based on [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3).

### Customize zones

We keep the main body of mmdetection unchanged. All the functions for zone evaluation are copied from the original ones.
For examples, 

`get_bboxes()` $\rightarrow$ `get_zone_bboxes()`, 

`_get_bboxes_single()` $\rightarrow$ `_get_zone_bboxes_single()`,

`_bbox_post_process()` $\rightarrow$ `_zone_bbox_post_process()`.

Please refer to [base_dense_head.py](mmdet/models/dense_heads/base_dense_head.py).

In [_zone_bbox_post_process](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/models/dense_heads/base_dense_head.py#L507),
we define zones as a series of annular regions. You can define the zone shape to be whatever you want.

Then, in [single_stage.py](mmdet/models/detectors/single_stage.py), it recieves 5 zone results.

In [multi_gpu_test](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/apis/test.py#L81),
it packs up the detection results of 5 zones.

In [test.py](tools/test.py), the function `zone_evaluate()` evaluates the zone results.

For VOC, the ignored ground-truth boxes are defined in [mean_ap.py](https://github.com/Zzh-tju/ZoneEval/blob/main/mmdetection/mmdet/core/evaluation/mean_ap.py#L265). For MS COCO, it lies in [cocoeval.py](https://github.com/Zzh-tju/ZoneEval/blob/main/pycocotools/pycocotools/cocoeval.py#L596).
