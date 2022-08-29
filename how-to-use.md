# Zone Evaluation and Spatial Equilibrium Label Assignment

This page is the tutorial to our CVPR 2023, created by [Zhaohui Zheng](https://github.com/Zzh-tju).

## Note
### Chapter 1 -- [Introduction](https://github.com/Zzh-tju/SELA/blob/main/how-to-use.md#introduction)
### Chapter 2 -- Zone measures
### Chapter 3 -- SELA

# Introduction

In decades development of object detection, the evaluation is conducted in the full-map zone.
The detector produces N boxes after NMS, and then feeds them into evaluation process. 
This evaluation paradigm is dominant for years, which is standard, popular, and adopted in all the existing detection datasets, like PASCAL VOC, MS COCO, etc.
Is it reliable? Is it able to reflect the real performance of an object detector? We wonder.
From a historical point of view, the evaluation of object detection is a continual practice of that for image classification, that the evaluation always considers all the detections and Ground-truth (GT) boxes in the whole image zone.
This is actually based on a hypothesis that the detector can performs uniformly in an image, i.e., the detector can produce almost the same predictions for a given object no matter where it is.
Is it possibily achievable? Emm, maybe, but still hard.
In fact, the detection performance is highly related to the data distribution.
For CNN-based dense detectors, a zone with more objects means it will recieve much more supervision signals during training, and therefore be endowed much better detection ability.
While the detection performance will be poor in those zones with less frequency of objects.
Owing to the photographer's bias, the center zone of the image usually has the most number of objects.
From this, we find something amazing that the current evaluation is actually affected by the extreme performance in a tiny zone, i.e., the center zone, commonly.
