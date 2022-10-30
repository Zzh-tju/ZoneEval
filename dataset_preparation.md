## Dataset Preparations

### Get The PASCAL VOC Dataset:

We use VOC 07+12 protocol, i.e., the train set contains VOC 2007 trainval + VOC 2012 trainval, and the test set contains VOC 2007 test.

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

The dowoloaded datasets may have some errors:

 - For face mask dataset, we use the union of train set and valid set for training.
 - For fruit dataset, we found some images have errors. Please use our `train.txt and `test.txt` in `data/fruit/VOC2007/ImageSets/Main` to process the normal data.
 - For helmet dataset, we use the union of valid set and test set for evaluation.

### Get The MS COCO Dataset:

We use COCO train2017 for training and val2017 for evaluation.

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

The breif information about the 5 datasets:

| Dataset | Training set | Test set | #Classes |
|----------|:--------:|:--------:|:--------:|
| PASCAL VOC | 16551 | 4952 | 20 |
| Face Mask | 5865 | 1035 | 2 |
| Fruit | 3836 | 639 | 11 |
| Helmet | 15887 | 6902 | 2 |
| MS COCO | 118K | 5K | 80 |
