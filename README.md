<img src="flyleaf.png"/>

### This repo is based on [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection) 

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

### Here is a detailed step-by-step [tutorials](https://github.com/Zzh-tju/SELA/blob/main/tutorial.md).

