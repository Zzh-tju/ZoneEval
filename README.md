<img src="tutorials/flyleaf.png"/>

### This repo is based on [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection) 

### Here is a detailed step-by-step [tutorial](tutorials/readme.md).

This is the source codes of our paper.

```
@article{zheng2023ZoneEval,
  title={Towards Spatial Equilibrium Object Detection},
  author= {Zheng, Zhaohui and Chen, Yuming and Hou, Qibin and Li, Xiang and Cheng, Ming-Ming},
  journal={arxiv},
  year={2023}
}
```


## Installation

```
conda create --name ZoneEval python=3.8 -y

conda activate ZoneEval

conda install pytorch=1.12 cudatoolkit=11.3 torchvision=0.13.0 -c pytorch

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

git clone https://github.com/Zzh-tju/ZoneEval.git

cd ZoneEval/pycocotools

pip install -e .

cd ..

cd mmdetection

pip install -v -e .
```

## Dataset Preparations

Please refer to [Dataset Preparations](https://github.com/Zzh-tju/SELA/blob/main/dataset_preparation.md) for preparing PASCAL VOC 07+12, Face Mask, Fruit, Helmet, and MS COCO datasets.

## Evaluation

### Turn on zone evaluation

The relevant options can be specified on the config file,

```
model = dict(
    test_cfg=dict(zone_eval=True))   # set to False and evaluate in the conventional way.
```

### Evaluation command

```
# for VOC and 3 application datasets

./tools/dist_test.sh configs/sela/your_config_file.py your_model.pth 2 --eval mAP

# for MS COCO

./tools/dist_test.sh configs/sela/your_config_file.py your_model.pth 2 --eval bbox
```

Currently, we provide evaluation for various object detectors, and the pretrained weight file can be downloaded from MMDetection or their official websites.

| Detector | Network & TS | SP | $\text{ZP}^{0,5}$| Variance | $\text{ZP}^{0,1}$ | $\text{ZP}^{1,2}$ | $\text{ZP}^{2,3}$ | $\text{ZP}^{3,4}$ | $\text{ZP}^{4,5}$ |
|----------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) | R50_1x | 32.0 | 36.5 | 14.8 | 27.3 | 33.3 | 35.5 | 34.5 | 39.2 |
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) | R50_2x | 32.6 | 37.4 | 16.9 | 27.6 | 34.6 | 35.8 | 35.1 | 40.4 |
|[Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) | R50_1x | 33.1 | 37.4 | 11.8 | 29.3 | 34.2 | 36.1 | 35.0 | 39.9 |
|[YOLOF](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof) | R50_1x | 33.2 | 37.5 | 12.8 | 28.4 | 35.2 | 36.6 | 35.3 | 39.2 |
|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn) | R50_1x | 33.3 | 37.9 | 22.8 | 27.8 | 34.7 | 37.1 | 37.1 | 42.6 |
|[YOLOv5-s](https://github.com/ultralytics/yolov5) | | 33.3 | 37.4 | 10.5 | 28.8 | 34.9 | 36.9 | 35.1 | 38.4 |
|[RepPoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints) | R50_1x | 33.5 | 38.1 | 12.9 | 29.2 | 34.7 | 36.7 | 35.6 | 40.3 |
|[FCOS](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos) | R50_1x | 34.2 | 38.7 | 14.7 | 29.5 | 35.3 | 38.0 | 36.7 | 41.1 |
| |
|[DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr) | R50_150e | 35.3 | 40.1 | 26.9 | 29.8 | 36.2 | 39.8 | 39.1 | 45.7 |
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt) | PVT-s_1x | 35.5 | 40.4 | 19.7 | 30.8 | 36.9 | 39.0 | 37.4 | 44.6 |
[Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) | R50_1x | 35.6 | 40.3 | 18.7 | 30.9 | 36.6 | 39.2 | 38.6 | 44.2 |
|[GFocal](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl) | R50_1x | 35.7 | 40.1 | 14.4 | 30.9 | 36.6 | 39.2 | 38.6 | 44.2 |
| |
|[YOLOv8-s](https://github.com/ultralytics/ultralytics) | | 39.8 | 44.9 | 24.4 | 33.4 | 42.2 | 44.3 | 43.2 | 48.5 |
|[Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) | R101_3x | 40.3 | 45.4 | 22.4 | 34.7 | 41.6 | 44.3 | 44.4 | 49.1 |
|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn) | R50_3x | 40.6 | 45.0 | 21.6 | 35.8 | 41.9 | 43.4 | 44.0 | 50.3 |
|[YOLOv5-m](https://github.com/ultralytics/yolov5) | | 40.8 | 45.2 | 12.9 | 36.0 | 42.3 | 44.5 | 43.2 | 46.7 |
| |
[Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin) | Swin-T_3x | 40.9 | 46.0 | 15.4 | 36.8 | 41.7 | 44.1 | 43.5 | 49.0 |
|[Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/convnext) | ConvNeXt-T_3x | 41.1 | 46.2 | 17.6 | 46.7 | 41.9 | 44.5 | 43.6 | 49.7 |
|[Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) | X101-32x8d_3x | 41.2 | 46.1 | 21.1 | 36.1 | 42.0 | 44.8 | 45.9 | 49.9 |
|[VFNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/vfnet) | R101_2x | 41.5 | 46.2 | 15.6 | 36.7 | 43.0 | 45.0 | 44.5 | 48.8 |
|[Deformable DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr) | R50_50e | 41.6 | 46.1 | 23.2 | 36.3 | 42.6 | 45.6 | 45.1 | 51.2|
|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn) | R101_3x | 41.7 | 46.2 | 21.1 | 36.9 | 42.9 | 44.9 | 44.7 | 51.3 |
|[GFocal](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl) | X101-32x4d_2x | 41.8 | 46.1 | 15.7 | 37.0 | 43.5 | 45.0 | 44.4 | 49.3 |

#### Note: 
 - 'TS': Training Schedule. 
 - ' $\text{ZP}^{0,5}$ ': the traditional AP.
 - 'Variance': the variance of the 5 ZPs ( $\text{ZP}^{0,1}$, $\text{ZP}^{1,2}$, ..., $\text{ZP}^{4,5}$ ).
 - If you test DETR series, you must modify the `simple_test()` function in `mmdet/models/detectors/single_stage.py`,

```python
        #outs = self.bbox_head(feat)
        outs = self.bbox_head(feat, img_metas) # if you test DETR series
```

Currently, we do not support zone evaluation for instance segmentation models.

