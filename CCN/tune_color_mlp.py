import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from torch import nn
# from tensorboard_logger import configure, log_value
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from modules import ColorMLP, GATLoss
from util import save_checkpoint, accuracy, save_yaml_file

warnings.filterwarnings("ignore")


def main(args):
    assert len(args.l) == 3, '--l must be 3 float numbers'
    # print(type(args.l))
    # print(args.l)
    log_dir = args.log_dir if args.log_dir[-1] == '/' else args.log_dir + '/'
    log_dir += datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_yaml_file(log_dir, args, 'config.yml')

    data = np.load(args.log_data)
    X = data[:, 0:3]
    y = data[:, 3]
    # t = data[:, 4]
    classes = config['classes']

    # print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    # print(f'target classes: {classes}')

    adjs = config['rgb_channel_adj']

    # normalize data
    mean = np.array(config['mean'])
    std = np.array(config['std'])

    # print(f'before normalization: mean={X.mean()}, std={X.std()}')
    X = (X - mean) / std
    # print(f'after normalization: mean={X.mean()}, std={X.std()}')

    num_classes = len(classes)
    num_features = 3

    # shuffle = False
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=False)

    pred_y = np.array([])
    true_y = np.array([])
    logits = []
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        current_log_dir = log_dir + f'/fold_{i}'
        if not os.path.exists(current_log_dir):
            os.makedirs(current_log_dir)

        # np.save(f'{current_log_dir}/train_indices.npy', train_indices)
        # np.save(f'{current_log_dir}/test_indices.npy', test_indices)

        # configure(log_dir)
        writer = SummaryWriter(log_dir=current_log_dir)

        print(f'fold {i}:')
        train_X, train_y, test_X, test_y = X[train_indices], y[train_indices], X[test_indices], y[test_indices]

        true_y = np.append(true_y, test_y)

        train_X, train_y, \
        test_X, test_y = torch.from_numpy(train_X).to(args.device).float(), torch.from_numpy(train_y).to(
            args.device).long(), \
                         torch.from_numpy(test_X).to(args.device).float(), torch.from_numpy(test_y).to(
            args.device).long()

        network = ColorMLP(input_dim=num_features, hidden_dim=args.hidden_dim, output_dim=num_classes,
                           num_heads=args.num_heads, alpha=args.alpha, adjs=adjs.to(args.device)).to(args.device)

        criterion = nn.CrossEntropyLoss().to(args.device)
        g_criterion = GATLoss(l=torch.tensor(args.l).unsqueeze(0).to(args.device),
                              adjs=adjs.to(args.device)).to(args.device)
        optimizer = torch.optim.Adam(params=network.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in tqdm(range(args.epochs)):
            # shuffle
            train_X, train_y = shuffle(train_X, train_y, random_state=0)

            train(train_X, train_y, network, criterion, g_criterion, optimizer, writer, epoch)
            test_acc = test(test_X, test_y, network, criterion, g_criterion, writer, epoch)

            if epoch == args.epochs - 1:
                save_checkpoint(state={'epoch': epoch,
                                       'state_dict': network.state_dict(),
                                       'test_acc': test_acc},
                                saved_dir=current_log_dir,
                                filename='last.pth.tar')
                with torch.no_grad():
                    fold_logits = network(test_X)
                    logits.append(fold_logits)
                    fold_pred_y = (F.softmax(fold_logits, dim=1)).argmax(dim=1).cpu().numpy()
                    pred_y = np.append(pred_y, fold_pred_y)

        writer.close()

    # confusion matrix
    cm = confusion_matrix(y_true=true_y, y_pred=pred_y)
    np.save(f'{log_dir}/cm.npy', cm)

    # save config
    result_dict = {}
    logits = torch.cat(logits, dim=0).cpu()
    top1_acc, top5_acc = accuracy(logits.log_data, torch.from_numpy(true_y), topk=(1, 5))
    result_dict['top1_acc'], result_dict['top5_acc'] = top1_acc.item(), top5_acc.item()
    save_yaml_file(log_dir, result_dict, 'result.yml')

    print(result_dict)


def train(train_X, train_y, network, criterion, g_criterion, optimizer, writer, epoch):
    network.train()
    logits = network(train_X)
    prec1, prec5 = accuracy(logits.log_data, train_y, topk=(1, 5))
    cel = criterion(logits, train_y)
    gl = g_criterion(network.get_w())
    loss = cel + gl

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log
    writer.add_scalar('train/top1_acc', prec1.item(), epoch)
    writer.add_scalar('train/top5_acc', prec5.item(), epoch)
    writer.add_scalar('train/loss', loss.item(), epoch)
    writer.add_scalar('train/cel', cel.item(), epoch)
    writer.add_scalar('train/gl', gl.item(), epoch)


def test(test_X, test_y, network, criterion, g_criterion, writer, epoch):
    network.eval()
    logits = network(test_X)
    prec1, prec5 = accuracy(logits.log_data, test_y, topk=(1, 5))
    cel = criterion(logits, test_y)
    gl = g_criterion(network.get_w())
    loss = cel + gl

    # log
    writer.add_scalar('test/top1_acc', prec1.item(), epoch)
    writer.add_scalar('test/top5_acc', prec5.item(), epoch)
    writer.add_scalar('test/loss', loss.item(), epoch)
    writer.add_scalar('test/cel', cel.item(), epoch)
    writer.add_scalar('test/gl', gl.item(), epoch)

    return prec1.item()


if __name__ == '__main__':
    """
    python tune_color_mlp.py --data ./data/data.npy --log_dir ./runs/tune/color_mlp --l 1e-05 1e-05 0.01
    """

    parser = argparse.ArgumentParser(description='tuning ColorMLP for Computational Color Naming.')

    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device, cuda or cuda:i or cpu, default: cuda:0')

    parser.add_argument('--hidden_dim', default=18, type=int,
                        help='numbers of hidden dim, default: 18')

    parser.add_argument('--num_heads', default=5, type=int,
                        help='numbers of heads in GALs in ColorMLP, default: 5')

    parser.add_argument('--alpha', default=0.2, type=float,
                        help='alpha for leaky relu activation function, default: 0.2')

    parser.add_argument('--l', default=[0.0001, 0.001, 1e-05], type=float, nargs='+',
                        help='hyper-parameter lambda for GATLoss component, default: [0.0001, 0.001, 1e-05]')

    parser.add_argument('--data', default='./data/data.npy',
                        help='the training data.')

    parser.add_argument('--num_folds', default=5, type=int,
                        help='numbers of folds for cross-validation, default: 5')

    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr',
                        help='initial learning rate, default: 0.1')

    parser.add_argument('--wd', '--weight-decay', default=0, type=float, dest='wd',
                        help='weight decay, default: 0')

    parser.add_argument('--log_dir', default='./runs/tune/color_mlp', type=str,
                        help='the directory where checkpoint and logs to be saved, default: ./runs/tune/color_mlp')

    args = parser.parse_args()
    main(args)
