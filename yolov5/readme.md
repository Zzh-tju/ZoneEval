### Zone Evaluation on YOLOv5

Download YOLOv5 from https://github.com/ultralytics/yolov5 and install the dependencies. **As a reminder, we use [YOLOv5-v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0).**

Make sure you have compiled our [pycocotools](https://github.com/Zzh-tju/SELA#installation).

Download the pretrained weight file, e.g., YOLOv5-s, from the official website.

Put our provided file `zone-val.py` to `./yourpath/yolov5/`.

Then run,

```python
 python zone-val.py --data coco.yaml --img 640  --weight yolov5s.pt --device 0,1
```

And you will see the zone evaluation results (YOLOv5-s):

```python
Zone:, ZP, ZP50, ZP75, ZPs, ZPm, ZPl, ZR1, ZR10, ZR100, ZRs, ZRm, ZRl
z05:  [       37.4        57.2        40.2        21.1        42.3          49        31.1        51.6        56.6        37.8        62.5        72.2]
---------------------------------------
z01:  [       28.8        44.3        30.4        19.4        36.8        42.7        34.3        49.4        51.3        40.6        60.5        66.3]
z12:  [       34.9        52.7          38        21.6        41.3        46.3        37.7        53.2        55.1        36.8        60.2        70.8]
z23:  [       36.9          55        39.3        20.7        42.7        49.1        37.2        52.1        53.9        36.8        59.7        67.4]
z34:  [       35.1        53.8        37.2        21.8        41.6        45.9        37.7        50.5        52.1        35.6        58.6        63.2]
z45:  [       38.4        57.8        42.3          27        45.1        47.6        45.6        54.9          56        39.2        61.1          69]
---------------------------------------
ZP_variance:  [     10.488      20.739       15.42      6.6149      7.2542      4.5353      14.179      3.8022      3.1174       3.321     0.69311      6.5777]
SP, SP50, SP75, SPs, SPm, SPl, SR1, SR10, SR100, SRs, SRm, SRl
SP: [       33.3        50.5        35.6        20.9        40.1        45.6        36.7        51.3        53.2        38.1        60.1        67.5]
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
