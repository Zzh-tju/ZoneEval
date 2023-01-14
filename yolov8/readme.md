### Zone Evaluation on YOLOv8

Download YOLOv8 from https://github.com/ultralytics/ultralytics and install the dependencies.

Make sure you have compiled our [pycocotools](https://github.com/Zzh-tju/SELA#installation).

Replace `./yourpath/ultralytics/ultralytics/yolo/v8/detect/val.py` with our provided file `val.py`.

You have to install numpy-1.23.3 to avoid error.

```python
conda install numpy=1.23.3
```

Recompile

```python
 pip install -e '.[dev]'
```

Then run,

```python
yolo mode=val task=detect data=coco.yaml model=yolov8n.pt device=\'0,1\'
```

And you will see the zone evaluation results (YOLOv8-n):

```python
Zone:, ZP, ZP50, ZP75, ZPs, ZPm, ZPl, ZR1, ZR10, ZR100, ZRs, ZRm, ZRl
z05:  [       37.3        52.5        40.5        18.5          41        53.5          32        53.2        58.8        36.6        65.3        76.8]
---------------------------------------
z01:  [       26.4        37.5        28.6        15.7        34.4        45.2        33.8        48.8        50.8        35.3        62.3        69.9]
z12:  [         34        47.9        36.6        17.4        41.1        48.4        38.7        54.7        56.9        34.9        62.7        73.9]
z23:  [         37        51.2        39.8        18.2        41.8        52.9        38.2        55.1        57.2          37        64.6        72.3]
z34:  [       35.8        50.5        38.4        19.8        41.5        52.5        39.2          53        54.9        34.4        63.4        70.3]
z45:  [       39.6        54.7        43.3        24.6          43        54.5        48.3        57.9        58.7        38.1        63.9        76.8]
---------------------------------------
ZP_variance:  [     19.717      34.237       24.03      9.1978      9.3539      11.555      22.441      9.0023      7.3606      1.9472     0.68575       6.403]
SP, SP50, SP75, SPs, SPm, SPl, SR1, SR10, SR100, SRs, SRm, SRl
SP: [       32.3        45.4        34.8        17.5        38.9        48.9        37.3        52.6        54.6        35.5          63        71.8]
```

Test resolution: 640.

| Detector | SP | $\text{ZP}^{0,5}$| Variance | $\text{ZP}^{0,1}$ | $\text{ZP}^{1,2}$ | $\text{ZP}^{2,3}$ | $\text{ZP}^{3,4}$ | $\text{ZP}^{4,5}$ | params<br><sup>(M) | FLOPs<br><sup>(B) |
|----------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|[YOLOv5-s](https://github.com/ultralytics/yolov5) | 33.3 | 37.4 | 10.5 | 28.8 | 34.9 | 36.9 | 35.1 | 38.4 | 7.2 | 16.5 |
|[YOLOv8-n](https://github.com/ultralytics/ultralytics) | 32.3 | 37.3 | 19.7 | 26.4 | 34.0 | 37.0 | 35.8 | 39.6 | 3.2 | 8.7 |
| |
|[YOLOv5-m](https://github.com/ultralytics/yolov5) | 40.8 | 45.2 | 12.9 | 36.0 | 42.3 | 44.5 | 43.2 | 46.7 | 21.2 | 49.0 |
|[YOLOv8-s](https://github.com/ultralytics/ultralytics) | 39.8 | 44.9 | 24.4 | 33.4 | 42.2 | 44.3 | 43.2 | 48.5 | 11.2 | 28.6 |

 **Discussion**: If we compare YOLOv8 and YOLOv5 with similar AP, the improvement of YOLOv8 mainly comes from large objects and central zone. Besides, YOLOv5 performs better in spatial equilibrium (lower variance).
