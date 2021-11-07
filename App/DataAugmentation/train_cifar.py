import argparse
import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value
from tqdm import tqdm

from pcj import PartialColorJitter
from util import save_yaml_file, accuracy, AverageMeter, save_checkpoint

"""
Training Configurations

CIFAR-10/100:

DenseNet refers to: [paper]Densely Connected Convolutional Networks
                    [code]https://github.com/bamos/densenet.pytorch/blob/master/train.py
                    [paper]Regularizing deep networks with semantic data augmentation
                    [code]https://github.com/blackfeather-wang/ISDA-for-Deep-Networks/blob/master/Image%20classification%20on%20CIFAR/train.py

ResNet refers to: [paper]Regularizing deep networks with semantic data augmentation
                  [code]https://github.com/blackfeather-wang/ISDA-for-Deep-Networks/blob/master/Image%20classification%20on%20CIFAR/train.py

Inception refers to: [paper]Rethinking the Inception Architecture for Computer Vision
                     [code]https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/train.py

"""
training_config = {
    'densenet': {
        'optimizer': {'type': 'SGD',
                      'lr': 0.1,
                      'momentum': 0.9,
                      'wd': 1e-4,
                      'nesterov': True},
        'lr_scheduler': {'type': 'MultiStepLR',
                         'step': 'epoch',
                         'milestones': [150, 200, 250],
                         'gamma': 0.1},
        'epochs': 300,
        'batch_size': 64
    },
    'inception': {
        'optimizer': {'type': 'SGD',
                      'lr': 1e-2,
                      'momentum': 0.9,
                      'wd': 1e-2,
                      'nesterov': True},
        'lr_scheduler': {'type': 'WarmupCosineLR',
                         'step': 'iter',
                         'warmup_rate': 0.3},
        'epochs': 100,
        'batch_size': 256  # = 256 your GPU memory must be greater equal than 16GB, or you can let it be 128.
    },
    'resnet': {
        'optimizer': {'type': 'SGD',
                      'lr': 0.1,
                      'momentum': 0.9,
                      'wd': 1e-4,
                      'nesterov': True},
        'lr_scheduler': {'type': 'MultiStepLR',
                         'step': 'epoch',
                         'milestones': [80, 120],
                         'gamma': 0.1},
        'epochs': 160,
        'batch_size': 128
    }
}


def main(args):
    global training_config
    # print('config:')
    # for k in args.__dict__:
    #     print(f'{k}: {args.__dict__[k]}')

    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # log_dir = args.log_dir if args.log_dir[-1] == '/' else args.log_dir + '/'
    # log_dir += datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    log_dir = os.path.join(args.log_dir, args.datetime)
    # print(log_dir)
    configure(log_dir)
    save_yaml_file(log_dir, args, 'config.yml')

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise Exception(f'unsupported dataset type: {args.dataset}!')

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

    # read config
    arch = args.arch
    if arch.startswith('densenet'):
        arch = 'densenet'
    elif arch.startswith('inception'):
        arch = 'inception'
    elif arch.startswith('resnet'):
        arch = 'resnet'
    else:
        raise Exception('unsupported type: {}'.format(arch))

    config = training_config[arch]

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.__dict__[args.dataset.upper()](root=args.dataset_dir, train=True, download=True,
                                                            transform=train_transform),
        batch_size=config['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.__dict__[args.dataset.upper()](root=args.dataset_dir, train=False, download=True,
                                                            transform=test_transform),
        batch_size=config['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # network = eval(args.arch)(num_classes).to(args.device)
    network = getattr(__import__('networks'), args.arch)(num_classes).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if config['optimizer']['type'] == 'Adadelta':
        optimizer = getattr(torch.optim, config['optimizer']['type'])(params=network.parameters(),
                                                                      lr=config['optimizer']['lr'],
                                                                      weight_decay=config['optimizer']['wd'],
                                                                      rho=config['optimizer']['rho'],
                                                                      eps=config['optimizer']['eps'])
    elif config['optimizer']['type'] == 'SGD':
        optimizer = getattr(torch.optim, config['optimizer']['type'])(params=network.parameters(),
                                                                      lr=config['optimizer']['lr'],
                                                                      weight_decay=config['optimizer']['wd'],
                                                                      momentum=config['optimizer']['momentum'],
                                                                      nesterov=config['optimizer']['nesterov'])
    else:
        raise Exception('unsupported optimizer type!')

    # lr_scheduler step type: epoch or iter
    if config['lr_scheduler']['step'] == 'epoch':
        lr_step_on_each_epoch_or_iter = True

    elif config['lr_scheduler']['step'] == 'iter':
        lr_step_on_each_epoch_or_iter = False
    else:
        raise Exception('unsupported lr_scheduler step: {0} type.'.format(config['lr_scheduler']['step']))

    if config['lr_scheduler']['type'] == 'MultiStepLR':
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               config['lr_scheduler']['type'])(optimizer=optimizer,
                                                               milestones=config['lr_scheduler']['milestones'],
                                                               gamma=config['lr_scheduler']['gamma'])
    elif config['lr_scheduler']['type'] == 'CosineAnnealingLR':
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               config['lr_scheduler']['type'])(optimizer=optimizer,
                                                               T_max=config['lr_scheduler']['T_max'])
    elif config['lr_scheduler']['type'] == 'WarmupCosineLR':
        # InceptionV3
        lr_scheduler = getattr(__import__('scheduler'),
                               config['lr_scheduler']['type'])(optimizer=optimizer,
                                                               warmup_epochs=config['epochs']
                                                                             * len(train_loader)
                                                                             * config['lr_scheduler']['warmup_rate'],
                                                               max_epochs=config['epochs'] * len(train_loader))
    elif config['lr_scheduler']['type'] == 'CosineWarmupLR':
        # MobileNetV3
        lr_scheduler = getattr(__import__('scheduler'),
                               config['lr_scheduler']['type'])(optimizer=optimizer,
                                                               epochs=config['epochs'],
                                                               iter_in_one_epoch=len(train_loader),
                                                               lr_min=config['lr_scheduler']['lr_min'],
                                                               warmup_epochs=config['lr_scheduler']['warmup_epochs'])
    else:
        raise Exception('unsupported lr_scheduler type!')

    status = {'best_top1': {'acc': 0.0, 'epoch': 0},
              'best_top5': {'acc': 0.0, 'epoch': 0},
              'last': {'top1_acc': 0.0, 'top5_acc': 0, 'epoch': 0}}
    for epoch in tqdm(range(config['epochs'])):
        if lr_step_on_each_epoch_or_iter:
            train(train_loader, network, criterion, optimizer, None, epoch, args.device)
            test_top1_acc, test_top5_acc = validate(test_loader, network, criterion, epoch, args.device)
            # lr_scheduler step on each epoch
            lr_scheduler.step()
            log_value('lr', lr_scheduler.get_lr()[0], epoch)
        else:
            train(train_loader, network, criterion, optimizer, lr_scheduler, epoch, args.device)
            test_top1_acc, test_top5_acc = validate(test_loader, network, criterion, epoch, args.device)

        # save checkpoint
        is_best_test_top1 = test_top1_acc > status['best_top1']['acc']
        if is_best_test_top1:
            status['best_top1']['acc'] = test_top1_acc
            status['best_top1']['epoch'] = epoch
            save_checkpoint(state=network.state_dict(),
                            saved_dir=log_dir,
                            filename='best.pth.tar')

        is_best_test_top5 = test_top5_acc > status['best_top5']['acc']
        if is_best_test_top5:
            status['best_top5']['acc'] = test_top5_acc
            status['best_top5']['epoch'] = epoch

        if epoch == config['epochs'] - 1:
            status['last']['top1_acc'] = test_top1_acc
            status['last']['top5_acc'] = test_top5_acc
            status['last']['epoch'] = epoch
            save_checkpoint(state=network.state_dict(),
                            saved_dir=log_dir,
                            filename='last.pth.tar')

    print(status)

    # save result
    save_yaml_file(log_dir, status, 'result.yml')


def train(train_loader, network, criterion, optimizer, lr_scheduler, epoch, device):
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    losses = AverageMeter()

    network.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = network(inputs)

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        loss = criterion(outputs, targets)

        acc1.update(prec1.item(), inputs.size(0))
        acc5.update(prec5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr_scheduler step on each iter
        if lr_scheduler is not None:
            lr_scheduler.step()
            log_value('lr', lr_scheduler.get_lr()[0], epoch * len(train_loader) + i)

    # log
    log_value('train/top1_acc', acc1.avg, epoch)
    log_value('train/top5_acc', acc5.avg, epoch)
    log_value('train/loss', losses.avg, epoch)


def validate(test_loader, network, criterion, epoch, device):
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    losses = AverageMeter()

    network.eval()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = network(inputs)

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        loss = criterion(outputs, targets)

        acc1.update(prec1.item(), inputs.size(0))
        acc5.update(prec5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

    # log
    log_value('test/top1_acc', acc1.avg, epoch)
    log_value('test/top5_acc', acc5.avg, epoch)
    log_value('test/loss', losses.avg, epoch)

    return acc1.avg, acc5.avg


if __name__ == '__main__':
    """

    # python train_cifar.py --arch vgg16 --dataset cifar10 --dataset_dir /root/Workspace/Dataset --augment rc_rhf_pcj --log_dir ./runs/cifar10/vgg16/rc_rhf_pcj
    
    """
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Image Classification on CIFAR-10/100 with DataAugmentation.')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device, cuda or cuda:i or cpu, default: cuda')

    parser.add_argument('--arch', type=str,
                        choices=['densenet_100_12', 'densenet_250_24', 'densenet_190_40',
                                 'inception_v3',
                                 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202'],
                        help='architecture')

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                        help='dataset: cifar10 or cifar100')

    parser.add_argument('--dataset_dir', default='./dataset', type=str,
                        help='dataset dir, default: ./dataset')

    parser.add_argument('--augment', type=str,
                        choices=['none', 'cj', 'pcj'],
                        help='data augmentation')

    parser.add_argument('--num_workers', default=16, type=int,
                        help='how many subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the main process. (default: 16)')

    parser.add_argument('--log_dir', default='./runs', type=str,
                        help='where checkpoint and logs to be saved, default: ./runs')

    args = parser.parse_args()

    main(args)
