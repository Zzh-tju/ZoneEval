<img src="flyleaf.png"/>

### This repo is based on [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection) 

### Here is a detailed step-by-step [tutorial](https://github.com/Zzh-tju/SELA/blob/main/tutorial.md).

We provides the source code, evaluation protocols, and the tutorials of our paper.

```
@Inproceedings{zheng2023SELA,
  title={Towards Spatial Equilibrium Object Detection},
  author= {Zheng, Zhaohui and Chen, Yuming and Hou, Qibin and Li, Xiang and Cheng, Ming-Ming},
  booktitle={arxiv},
  year={2023}
}
```


## Installation

```
conda create --name SELA python=3.8 -y

conda activate SELA

conda install pytorch=1.12 cudatoolkit=11.3 torchvision=0.13.0 -c pytorch

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

git clone https://github.com/Zzh-tju/SELA.git

cd SELA/pycocotools

pip install -e .

cd ..

cd mmdetection

pip install -v -e .
```

## Datasets Preparation

### Get The PASCAL VOC Dataset:

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

### Get The 3 Application Datasets:


Download at Kaggle: [Face Mask](https://www.kaggle.com/datasets/parot99/face-mask-detection-yolo-darknet-format), [Fruit](https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection), [Helmet](https://www.kaggle.com/datasets/vodan37/yolo-helmethead).

The file dir follows the same settings to VOC-style.

```
cd data

mkdir mask

mkdir helmet

mkdir fruit
```

#### Convert to XML format if the data label is YOLO format.

Modify the file path to your datasets in `data/yolo2voc.py`, and run the command:

```
python data/yolo2voc.py
```

### Get The MS COCO Dataset:

```
python tools/misc/download_dataset.py --dataset-name coco2017
```

Put all the above datasets in the following dir
```
mmdetection
├── data
    ├── VOCdevkit
        ├── VOC2007
            ├──Annotations
            ├──ImageSets
            ├──JPEGImages
            ├──labels
            ├──SegmentationClass
            ├──SegmentationObject
        ├── VOC2012
            ├──Annotations
            ├──ImageSets
            ├──JPEGImages
            ├──labels
            ├──SegmentationClass
            ├──SegmentationObject
    ├── mask
        ├── VOC2007
            ├──Annotations
            ├──ImageSets
            ├──JPEGImages
    ├── fruit
        ├── VOC2007
            ├──Annotations
            ├──ImageSets
            ├──JPEGImages
    ├── helmet
        ├── VOC2007
            ├──Annotations
            ├──ImageSets
            ├──JPEGImages
    ├── coco
        ├── annotations
        ├── train2017
        ├── val2017
        ├── test2017
```

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

./tools/dist_test.sh configs/sela/gfl_r18_fpn_1x_voc.py your_model.pth 2 --eval mAP

# for MS COCO

./tools/dist_test.sh configs/sela/gfl_r50_fpn_1x_coco.py your_model.pth 2 --eval bbox
```

Currently, we provide evaluation for various object detectors, and the pretrained weight file can be downloaded from MMDetection or their official websites.

| Detector | Network | TS | SP | ZP@ $z_0^5$| Variance | ZP@ $z_0^1$ | ZP@ $z_1^2$ | ZP@ $z_2^3$ | ZP@ $z_3^4$ | ZP@ $z_4^5$ |
|----------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) | R50 | 1x | 32.0 | 36.5 | 14.8 | 27.3 | 33.3 | 35.5 | 34.5 | 39.2 |
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) | R50 | 2x | 32.6 | 37.4 | 16.9 | 27.6 | 34.6 | 35.8 | 35.1 | 40.4 |
|[Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) | R50 | 1x | 33.1 | 37.4 | 11.8 | 29.3 | 34.2 | 36.1 | 35.0 | 39.9 |
|[YOLOF](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof) | R50 | 1x | 33.2 | 37.5 | 12.8 | 28.4 | 35.2 | 36.6 | 35.3 | 39.2 |
|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn) | R50 | 1x | 33.3 | 37.9 | 22.8 | 27.8 | 34.7 | 37.1 | 37.1 | 42.6 |
|[RepPoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints) | R50 | 1x | 33.5 | 38.1 | 12.9 | 29.2 | 34.7 | 36.7 | 35.6 | 40.3 |
|[FCOS](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos) | R50 | 1x | 34.2 | 38.7 | 14.7 | 29.5 | 35.3 | 38.0 | 36.7 | 41.1 |
|[YOLOv5-s](https://github.com/ultralytics/yolov5) | | | 34.4 | 37.4 | 6.7 | 30.4 | 36.4 | 37.5 | 35.5 | 37.0 |
|[DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr) | R50 | 150e | 35.3 | 40.1 | 26.9 | 29.8 | 36.2 | 39.8 | 39.1 | 45.7 |
|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt) | PVT-Small | 1x | 35.5 | 40.4 | 19.7 | 30.8 | 36.9 | 39.0 | 37.4 | 44.6 |
[Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) | R50 | 1x | 35.6 | 40.3 | 18.7 | 30.9 | 36.6 | 39.2 | 38.6 | 44.2 |
|[GFocal](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl) | R50 | 1x | 35.7 | 40.1 | 14.4 | 30.9 | 36.6 | 39.2 | 38.6 | 44.2 |
[Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin) | Swin-T | 3x | 40.9 | 46.0 | 15.4 | 36.8 | 41.7 | 44.1 | 43.5 | 49.0 |
|[Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/convnext) | ConvNeXt-T | 3x | 41.1 | 46.2 | 17.6 | 46.7 | 41.9 | 44.5 | 43.6 | 49.7 |
|[Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) | X-101-32x8d-FPN | 3x | 41.2 | 46.1 | 21.1 | 36.1 | 42.0 | 44.8 | 45.9 | 49.9 |
|[VFNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/vfnet) | R101 | 2x | 41.5 | 46.2 | 15.6 | 36.7 | 43.0 | 45.0 | 44.5 | 48.8 |
|[Deformable DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr) | R50 | 50e | 41.6 | 46.1 | 23.2 | 36.3 | 42.6 | 45.6 | 45.1 | 51.2|
|[YOLOv5-m](https://github.com/ultralytics/yolov5) | | | 41.6 | 45.4 | 8.8 | 37.2 | 43.0 | 45.6 | 44.1 | 44.8 |
|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn) | R101 | 3x | 41.7 | 46.2 | 21.1 | 36.9 | 42.9 | 44.9 | 44.7 | 51.3 |

#### Note: 
 - 'TS': Training Schedule. 
 - 'ZP@ $z_0^5$': the traditional AP.
 - 'Variance': the variance of the 5 ZPs (ZP@ $z_0^1$, ZP@ $z_1^2$, ..., ZP@ $z_4^5$).
 - If you test DETR series, you must modify the `simple_test()` function in `mmdet/models/detectors/single_stage.py`,

```python
        #outs = self.bbox_head(feat)
        outs = self.bbox_head(feat, img_metas) # if you test DETR series
```

Currently, we do not support zone evaluation for instance segmentation models.

