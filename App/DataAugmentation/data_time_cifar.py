import argparse
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from pcj import PartialColorJitter
from util import AverageMeter


def main(args):
    print('config:')
    for k in args.__dict__:
        print(f'{k}: {args.__dict__[k]}')

    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # cj & pcj params
    brightness = 0.8
    contrast = 0.8
    saturation = 0.8
    hue = 0.2

    if args.augment == 'none':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment == 'cj':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=brightness,
                                   contrast=contrast,
                                   saturation=saturation,
                                   hue=hue),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment == 'pcj':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            PartialColorJitter(y_path='../../CCN/data_backup/y.npy',
                               brightness=brightness,
                               contrast=contrast,
                               saturation=saturation,
                               hue=hue),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise Exception('unsupported augment type!')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.__dict__[args.dataset.upper()](root=args.dataset_dir, train=True, download=True,
                                                            transform=train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.__dict__[args.dataset.upper()](root=args.dataset_dir, train=False, download=True,
                                                            transform=test_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    data_time = []
    batch_time = []
    for epoch in range(args.epochs):
        train_data_time, train_batch_time = train(train_loader)
        val_data_time, val_batch_time = validate(test_loader)
        data_time.append(train_data_time + val_data_time)
        batch_time.append(train_batch_time + val_batch_time)

    data_time = np.array(data_time) * 1000
    batch_time = np.array(batch_time) * 1000

    print('data_time: {} ± {} ms'.format(data_time.mean(), data_time.std()))
    print('batch_time: {} ± {} ms'.format(batch_time.mean(), batch_time.std()))


def train(train_loader):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for _ in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return data_time.avg, batch_time.avg


def validate(test_loader):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for _ in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return data_time.avg, batch_time.avg


if __name__ == '__main__':
    """
    # python /root/Workspace/CCN_and_ItsApp/App/DataAugmentation/wall_time_cifar.py --augment none --dataset cifar10 --dataset_dir /root/Workspace/Dataset
    """
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Data Loading Time on CIFAR-10/100 with DataAugmentation.')

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                        help='dataset: cifar10 or cifar100')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')

    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs')

    parser.add_argument('--dataset_dir', default='./dataset', type=str,
                        help='dataset dir, default: ./dataset')

    parser.add_argument('--augment', type=str,
                        choices=['none', 'cj', 'pcj'],
                        help='data augmentation')

    parser.add_argument('--num_workers', default=16, type=int,
                        help='how many subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the main process. (default: 16)')

    args = parser.parse_args()

    main(args)
