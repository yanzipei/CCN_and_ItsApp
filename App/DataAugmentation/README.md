# Note
## Image Classification on CIFAR-10/100:
network only be trained on a single instance with a single GPU support!

Supported Architectures:
inception_v3
resnet20, resnet32, resnet44, resnet56, resnet110
densenet_100_12

## Image Classification on ImageNet:
network only be trained on a single instance with multiply GPUs support!

Supported Architectures:
ResNet-50, ResNet-101, ResNet-152
DenseNet-BC-121

data_time_cifar.py: calculate data loading time per batch of different augmentations on CIFAR-10/100 dataset

data_time_imagenet.py: calculate data loading time per batch of different augmentations on ImageNet dataset

extract_ILSVRC.sh: extract ImageNet files

get_data_time.sh: bash script for executing data_time_cifar.py and data_time_imagenet.py

pcj.py: partial color jitter

read_log_cifar.py: read logs on CIFAR-10/100 datasets.

read_log_imagenet.py: read logs on ImageNet

run_cifar.sh: bash script for training different networks on CIFAR-10/100

run_imagenet.sh: bash script for training different networks on ImageNet

scheduler.py: learning rate scheduler

train_cifar.py: train different networks on CIFAR-10/100

train_imagenet.py: train different networks on ImageNet

util.py: common tool functions