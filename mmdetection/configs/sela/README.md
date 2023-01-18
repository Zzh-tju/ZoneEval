# SELA

> [Towards Spatial Equilibrium Object Detection](https://arxiv.org/abs/2301.05957)

<!-- [ALGORITHM] -->

## Abstract

Semantic objects are unevenly distributed over images.
In this paper, we study the spatial disequilibrium problem of modern object detectors and propose to quantify this ``spatial bias'' by measuring the detection performance over zones.
Our analysis surprisingly shows that the spatial imbalance of objects has a great impact on the detection performance, limiting the robustness of detection applications.
This motivates us to design a more generalized measurement, termed Spatial equilibrium Precision (SP), to better characterize the detection performance of object detectors.
Furthermore, we also present a spatial equilibrium label assignment (SELA) method
to alleviate the spatial disequilibrium problem by injecting the prior spatial weight into the optimization process of detectors.
Extensive experiments on PASCAL VOC, MS COCO, and 3 application datasets on face mask/fruit/helmet images demonstrate the advantages of our method.
Our findings challenge the conventional sense of object detectors and show the indispensability of spatial equilibrium.
We hope these discoveries would stimulate the community to rethink how an excellent object detector should be.
All the source code, evaluation protocols, and the tutorials will be made publicly available.

More details can be found in [tutorials](https://github.com/Zzh-tju/SELA/blob/main/tutorial.md).


## Citation

We provide config files to reproduce the object detection results in the paper.

```latex
@article{zheng2023ZoneEval,
  title={Towards Spatial Equilibrium Object Detection},
  author= {Zheng, Zhaohui and Chen, Yuming and Hou, Qibin and Li, Xiang and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2301.05957},
  year={2023}
}
```
