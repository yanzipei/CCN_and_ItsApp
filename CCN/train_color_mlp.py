import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorboard_logger import configure, log_value
from torch import nn
from tqdm import tqdm

from config import config
from modules import ColorMLP, GATLoss
from util import save_checkpoint, accuracy, save_yaml_file

warnings.filterwarnings("ignore")


def main(args):
    assert len(args.l) == 3, '--l must be 3 float numbers'
    # print(type(args.l))
    print(args.l)

    log_dir = args.log_dir if args.log_dir[-1] == '/' else args.log_dir + '/'
    log_dir += datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_yaml_file(log_dir, args, 'config.yml')
    configure(log_dir)

    data = np.load(args.data)
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

    X, y = torch.from_numpy(X).to(args.device).float(), torch.from_numpy(y).to(args.device).long()

    num_classes = len(classes)
    num_features = 3

    network = ColorMLP(input_dim=num_features, hidden_dim=args.hidden_dim, output_dim=num_classes,
                       num_heads=args.num_heads, alpha=args.alpha, adjs=adjs.to(args.device)).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    g_criterion = GATLoss(l=torch.tensor(args.l).unsqueeze(0).to(args.device),
                          adjs=adjs.to(args.device)).to(args.device)
    optimizer = torch.optim.Adam(params=network.parameters(), lr=args.lr, weight_decay=args.wd)

    status = {'top1_acc': 0.0, 'top5_acc': 0.0, 'epoch': 0}
    for epoch in tqdm(range(args.epochs)):
        # shuffle
        X, y = shuffle(X, y, random_state=0)

        status['top1_acc'], status['top5_acc'] = train(X, y, network, criterion, g_criterion, optimizer, epoch)

        if epoch == args.epochs - 1:
            status['epoch'] = epoch
            save_checkpoint(state=network.state_dict(),
                            saved_dir=log_dir,
                            filename='last.pth.tar')
            with torch.no_grad():
                pred_y = (F.softmax(network(X), dim=1)).argmax(dim=1)
                cm = confusion_matrix(y_true=y.cpu().numpy(), y_pred=pred_y.cpu().numpy())
                np.save(f'{log_dir}/cm.npy', cm)

    print(status)
    # save config
    save_yaml_file(log_dir, status, 'result.yml')


def train(train_X, train_y, network, criterion, g_criterion, optimizer, epoch):
    network.train()
    logits = network(train_X)
    prec1, prec5 = accuracy(logits.data, train_y, topk=(1, 5))
    cel = criterion(logits, train_y)
    gl = g_criterion(network.get_w())
    loss = cel + gl

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log
    log_value('train/top1_acc', prec1.item(), epoch)
    log_value('train/top5_acc', prec5.item(), epoch)
    log_value('train/loss', loss.item(), epoch)
    log_value('train/cel', cel.item(), epoch)
    log_value('train/gl', gl.item(), epoch)

    return prec1.item(), prec5.item()


if __name__ == '__main__':
    """
    python train_color_mlp.py --data ./data/data.npy --log_dir ./runs/train/color_mlp --hidden_dim 9 --l 0.0001 0.0001 0.001
    
    
    python train_color_mlp.py --data ./data/data2.npy --log_dir ./runs/train/color_mlp --hidden_dim 18 --l 0.0001 0.001 1e-05
    """

    parser = argparse.ArgumentParser(description='training ColorMLP for Computational Color Naming.')

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

    parser.add_argument('--data', default='./data/aug_data.npy',
                        help='the training data, default: ./data/aug_data.npy')

    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr',
                        help='initial learning rate, default: 0.1')

    parser.add_argument('--wd', '--weight-decay', default=0, type=float, dest='wd',
                        help='weight decay, default: 0')

    parser.add_argument('--log_dir', default='./runs/train/color_mlp', type=str,
                        help='the directory where checkpoint and logs to be saved, default: ./runs/train/color_mlp')

    args = parser.parse_args()
    main(args)
