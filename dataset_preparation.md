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

Download at our links: [Face Mask](https://drive.google.com/open?id=1fbBWeEKXrDt-hHHt3-7cJDojZR_ECi5w&authuser=0&usp=drive_link), [Fruit](https://drive.google.com/open?id=11DNfbHwbLP7Fg4wHwrh67sfZygsFKgx4&authuser=0&usp=drive_link), [Helmet](https://drive.google.com/open?id=1baU7YXZCpAYbw-ku05sro9oqvS1Lh70l&authuser=0&usp=drive_link).

Download at Kaggle: [Face Mask](https://www.kaggle.com/datasets/parot99/face-mask-detection-yolo-darknet-format), [Fruit](https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection), [Helmet](https://www.kaggle.com/datasets/vodan37/yolo-helmethead).

If you download at our links, you can use it after `unzip` and put them to the proper dir.

If you download at Kaggle links, you need to process the annotations to XML format if the data label is in YOLO format.

The dowoloaded datasets from kaggle may have some errors:

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
            ├── Annotations
                ├── 00001.xml
                    00002.xml
                    ......
            ├── ImageSets
                ├── Main
                    ├── train.txt
                        test.txt
            ├── JPEGImages
                ├── 00001.jpg
                    00002.jpg
                    ......
            ├── labels
            ├── SegmentationClass
            ├── SegmentationObject
        ├── VOC2012
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── labels
            ├── SegmentationClass
            ├── SegmentationObject
    ├── facemask
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
            ├── instances_train2017.json
                instances_val2017.json
        ├── train2017
            ├── 000000000009.jpg
                000000000025.jpg
                ......
        ├── val2017
            ├── 000000000139.jpg
                000000000285.jpg
                ......
        ├── test2017
```

The breif information about the 5 datasets:

| Dataset | Training set | Test set | #Classes |
|----------|:--------:|:--------:|:--------:|
| PASCAL VOC | 16551 | 4952 | 20 |
| Face Mask | 5865 | 1035 | 2 |
| Fruit | 3836 | 639 | 11 |
| Helmet | 15887 | 6902 | 2 |
| MS COCO | 118291 | 5000 | 80 |
