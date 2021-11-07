import argparse
import warnings
from datetime import datetime

import numpy as np
from joblib import dump
from sklearn import svm
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

from config import config
from util import save_yaml_file

warnings.filterwarnings("ignore")


def main(args):
    log_dir = args.log_dir if args.log_dir[-1] == '/' else args.log_dir + '/'
    log_dir += datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_yaml_file(log_dir, args, 'config.yml')
    # configure(log_dir)

    data = np.load(args.log_data)
    X = data[:, 0:3]
    y = data[:, 3]
    # t = data[:, 4]
    # classes = config['classes']

    # print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    # print(f'target classes: {classes}')

    # normalize data
    mean = np.array(config['mean'])
    std = np.array(config['std'])

    # print(f'before normalization: mean={X.mean()}, std={X.std()}')
    X = (X - mean) / std
    # print(f'after normalization: mean={X.mean()}, std={X.std()}')

    # X, y = torch.from_numpy(X).to(args.device).float(), torch.from_numpy(y).to(args.device).long()

    # num_classes = len(classes)
    # num_features = 3

    clf = svm.SVC(C=args.C, probability=True)
    clf = clf.fit(X, y)
    # save
    dump(clf, f'{log_dir}/clf.joblib')
    # from joblib import dump, load
    # clf = load('clf.joblib')
    # predict
    pred_y = clf.predict(X)
    score_y = clf.predict_proba(X)
    print(pred_y.shape)
    print(score_y.shape)

    status = {'top1_acc': 0.0, 'top5_acc': 0.0}
    top1_acc = top_k_accuracy_score(y, score_y, k=1)
    top5_acc = top_k_accuracy_score(y, score_y, k=5)
    status['top1_acc'], status['top5_acc'] = top1_acc, top5_acc

    cm = confusion_matrix(y_true=y, y_pred=pred_y)
    np.save(f'{log_dir}/cm.npy', cm)

    print(status)
    # save config
    save_yaml_file(log_dir, status, 'result.yml')


if __name__ == '__main__':
    """
    python train_svm.py --data ./data/data.npy --log_dir ./runs/train/svm
    python train_svm.py --data ./data/data_except_bgw_400.npy --log_dir ./runs/train/svm
    """

    parser = argparse.ArgumentParser(description='training SVM for Computational Color Naming.')

    parser.add_argument('--C', default=8000.0, type=float,
                        help='Regularization parameter. '
                             'The strength of the regularization is inversely proportional to C. '
                             'Must be strictly positive. The penalty is a squared l2 penalty.')

    parser.add_argument('--data', default='./data/aug_data.npy', type=str,
                        help='the training data, default: ./data/aug_data.npy')

    parser.add_argument('--log_dir', default='./runs/train/svm', type=str,
                        help='the directory where checkpoint and logs to be saved, default: ./runs/train/svm')

    args = parser.parse_args()
    main(args)
