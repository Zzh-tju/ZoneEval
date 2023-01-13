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

And you will see the zone evaluation results (YOLOv5-n):

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
