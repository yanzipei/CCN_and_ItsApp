# Experiments on Image Classification

## Introduction
This repository contains the code for experiments on image classification.

## Usage
### Init
Download CIFAR-10 and CIFAR-100 datasets. \
Download ImageNet, use this script to extract data: `https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4`

### Image Classification on CIFAR-10/100:
Network is trained on a single instance with a single GPU support!

Networks: 
inception_v3, 
resnet20, resnet32, resnet44, resnet56, resnet110 \
densenet_100_12

Datasets: 
cifar10, 
cifar100

Augmentation: 
none, 
cj, 
pcj

e.g., train resnet20 on cifar10 with pcj augmentation, other settings and parameters are the default.
```
python train_cifar.py --arch resnet10 --dataset cifar10 --dataset_dir ./Dataset --augment pcj --log_dir ./runs/cifar10/resnet/pcj
```

### Image Classification on ImageNet:

Network is trained on a single instance with multiply GPUs support!

Networks: 
ResNet-50, ResNet-101, ResNet-152 
DenseNet-BC-121

Augmentation: 
none, 
cj, 
pcj

e.g. train resnet50 on imagenet with pcj augmentation, other settings and parameters are the default.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_imagenet.py './Dataset/ImageNet_2012' --augment pcj --log_dir ./runs/imagenet/resnet50/pcj --arch resnet --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

## Structure
```text
.
├── README.md
├── data_time_cifar.py
├── data_time_imagenet.py
├── networks
│   ├── densenet.py
│   ├── inception.py
│   └── resnet.py
├── pcj.py
├── run_cifar.sh
├── run_imagenet.sh
├── scheduler.py
├── train_cifar.py
├── train_imagenet.py
└── util.py
```
## Description

`data_time_cifar.py`: calculate data loading time per batch of different augmentations on CIFAR-10/100 dataset.

`data_time_imagenet.py`: calculate data loading time per batch of different augmentations on ImageNet dataset.

`networks/desenet.py`: densenet for CIFAR-10/100.

`networks/inception.py`: inception for CIFAR-10/100.

`networks/resnet.py`: resnet for CIFAR-10/100.

`pcj.py`: partial color jitter.

`scheduler.py`: learning rate scheduler.

`train_cifar.py`: train different networks on CIFAR-10/100.

`train_imagenet.py`: train different networks on ImageNet.

`util.py`: common tool functions.