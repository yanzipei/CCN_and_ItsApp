archs=('resnet50' 'resnet101' 'resnet152'
  'inception_v3' 'densenet121')

augments=('pcj' 'none')

for arch in "${archs[@]}"; do
  for augment in "${augments[@]}"; do
    echo "arch: ${arch}, augment: ${augment}"
    # single node, multiple GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_imagenet.py '/root/Workspace/Dataset/06.ImageNet_2012' --augment "${augment}" --log_dir "./runs/imagenet/${arch}/${augment}" --arch "${arch}" --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
  done
done
