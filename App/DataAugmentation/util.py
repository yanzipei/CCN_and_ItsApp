import argparse
import os

import numpy as np
import torch
import yaml
from torch import nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, saved_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    saved_path = saved_dir + filename if saved_dir[-1] == '/' else saved_dir + '/' + filename
    torch.save(state, saved_path)


def save_yaml_file(dir, args, save_name='config.yml'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, save_name), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), percentage_or_decimal=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if percentage_or_decimal:
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(correct_k.mul_(1.0 / batch_size))
        return res


def print_num_params(net: nn.Module):
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print("Total layers: ", len(list(filter(lambda p: p.requires_grad and len(p.log_data.size()) > 1, net.parameters()))))
