### This is repo is based on [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3).

### Customize zones

We keep the main body of mmdetection unchanged. All the functions for zone evaluation are copied from the original ones.
For examples, 

`get_bboxes()` $\rightarrow$ `get_zone_bboxes()`, 

`_get_bboxes_single()` $\rightarrow$ `_get_zone_bboxes_single()`,

`_bbox_post_process()` $\rightarrow$ `_zone_bbox_post_process()`.

Please refer to [base_dense_head.py](mmdet/models/dense_heads/base_dense_head.py).
