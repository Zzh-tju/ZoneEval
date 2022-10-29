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

## Evaluation

### Turn on zone evaluation

The relevant options can be specified on the config file,

```
model = dict(
    test_cfg=dict(zone_measure=True))   # set to False and evaluate in the conventional way.
```

### Evaluation command

```
# for VOC and 3 application datasets

./tools/dist_test.sh configs/sela/gfl_r18_fpn_1x_voc.py your_model.pth 2 --eval mAP

# for MS COCO

./tools/dist_test.sh configs/sela/gfl_r50_fpn_1x_coco.py your_model.pth 2 --eval bbox
```

Currently, we provide evaluation for various object detectors, and the pretrained weight file can be downloaded from MMDetection or their official websites.

<div align="center">
  
|[Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)|[Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)|[RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet)|[FCOS](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos)|[RepPoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints)|
|-------|-------|-------|-------|-------|
|[DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr)|[Deformable DETR](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr)|[Sparse R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn)|[GFocal](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl)|[VFNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/vfnet)|
|[YOLOv5](https://github.com/ultralytics/yolov5)|[RetinaNet - Pyramid vision transformer](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt)|[Mask R-CNN - Swin Transformer](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin)|[Mask R-CNN - ConvNeXt](https://github.com/open-mmlab/mmdetection/tree/master/configs/convnext)| |
  
</div>

#### Note: if you test DETR series, you must modify the `simple_test()` function in `mmdet/models/detectors/single_stage.py`,

```python
        #outs = self.bbox_head(feat)
        outs = self.bbox_head(feat, img_metas) # if you test DETR series
```

Currently, we do not support zone evaluation for instance segmentation models.

