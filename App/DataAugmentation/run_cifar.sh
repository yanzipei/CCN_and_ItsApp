datasets=('cifar10' 'cifar100')

archs=('resnet20' 'resnet32' 'resnet44' 'resnet56' 'resnet110'
  'inception_v3'
  'densenet_100_12')

augments=('none' 'cj' 'pcj')

for dataset in "${datasets[@]}"; do
  for arch in "${archs[@]}"; do
    for augment in "${augments[@]}"; do
      echo "dataset: ${dataset}, arch: ${arch}, augment: ${augment}"
      python train_cifar.py --arch "${arch}" --dataset "${dataset}" --dataset_dir /root/Workspace/Dataset --augment "${augment}" --log_dir "./runs/${dataset}/${arch}/${augment}"
    done
  done
done
