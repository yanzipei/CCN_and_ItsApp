augments=('none' 'cj' 'pcj')

cifar_datasets=('cifar10' 'cifar100')

for dataset in "${cifar_datasets[@]}"; do
  for augment in "${augments[@]}"; do
    echo "${augment}, augment: ${augment}"
    python data_time_cifar.py --augment "${augment}" --dataset "${dataset}" --dataset_dir /root/Workspace/Dataset --epochs 100
  done
done


for augment in "${augments[@]}"; do
  echo "ImageNet, augment: ${augment}"
  # single node, multiple GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 python data_time_imagenet.py '/root/Workspace/Dataset/06.ImageNet_2012' --augment "${augment}" --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --epochs 20
done
