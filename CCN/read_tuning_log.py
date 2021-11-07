from glob import glob

import pandas as pd
import yaml


def read_color_mlp_tuning_log(log_dir, save_dir_and_name=None):
    dir = log_dir if log_dir[-1] == '/' else log_dir + '/'
    dir += '/*'
    data = []
    best_top1 = (0, None)
    best_top5 = (0, None)
    for dir0 in glob(dir + '*'):
        # print(dir0)
        datetime = dir0.split('/')[-1]
        lambdas, top1_acc, top5_acc = None, None, None
        for dir1 in glob(dir0 + '/*'):
            # print(dir1)
            filename = dir1.split('/')[-1]
            if filename == 'config.yml':
                # print(dir1)
                config = yaml.unsafe_load(open(dir1))
                lambdas = config.l

            if filename == 'result.yml':
                # print(dir1)
                result = yaml.unsafe_load(open(dir1))
                top1_acc = result['top1_acc']
                top5_acc = result['top5_acc']
                best_top1 = (top1_acc, lambdas) if top1_acc > best_top1[0] else best_top1
                best_top5 = (top5_acc, lambdas) if top5_acc > best_top5[0] else best_top5

        row = [lambdas, top1_acc, top5_acc, datetime]
        data.append(row)
    df = pd.DataFrame(data=data, columns=['lambda', 'top1_acc', 'top5_acc', 'datetime'])
    df = df.sort_values(by=['datetime'])
    print(df)

    print()

    print(f'best_top1_acc: {best_top1[0]}, lambdas: {best_top1[1]}')
    print(f'best_top5_acc: {best_top5[0]}, lambdas: {best_top5[1]}')

    if save_dir_and_name:
        df.to_csv(f'{save_dir_and_name}.csv')


def read_mlp_tuning_log(log_dir):
    dir = log_dir if log_dir[-1] == '/' else log_dir + '/'
    dir += '/*'
    data = []
    best_top1_acc = 0.0
    best_top5_acc = 0.0
    for dir0 in glob(dir + '*'):
        # print(dir0)
        datetime = dir0.split('/')[-1]
        # if not datetime.startswith('2021-06-23'):
        #     continue

        lr, wd, epochs, hidden_dim, top1_acc, top5_acc = None, None, None, None, None, None
        for dir1 in glob(dir0 + '/*'):
            # print(dir1)
            filename = dir1.split('/')[-1]
            if filename == 'config.yml':
                # print(dir1)
                config = yaml.unsafe_load(open(dir1))
                lr = config.lr
                wd = config.wd
                epochs = config.epochs
                hidden_dim = config.hidden_dim

            if filename == 'result.yml':
                # print(dir1)
                result = yaml.unsafe_load(open(dir1))
                top1_acc = result['top1_acc']
                top5_acc = result['top5_acc']
                best_top1_acc = top1_acc if top1_acc > best_top1_acc else best_top1_acc
                best_top5_acc = top5_acc if top5_acc > best_top5_acc else best_top5_acc

        # row = [lr, wd, epochs, hidden_dim, top1_acc, top5_acc, datetime]
        row = [lr, wd, epochs, hidden_dim, top1_acc, top5_acc]
        data.append(row)
    # df = pd.DataFrame(data=data, columns=['lr', 'wd', 'epochs', 'hidden_dim', 'top1_acc', 'top5_acc', 'datetime'])
    # df = df.sort_values(by=['datetime'])
    df = pd.DataFrame(data=data, columns=['lr', 'wd', 'epochs', 'hidden_dim', 'top1_acc', 'top5_acc'])
    print(df)

    print()

    print(best_top1_acc)
    print(best_top5_acc)


def read_svm_tuning_log(log_dir):
    dir = log_dir if log_dir[-1] == '/' else log_dir + '/'
    dir += '/*'
    data = []
    best_top1 = (0, None)
    best_top5 = (0, None)
    for dir0 in glob(dir + '*'):
        # print(dir0)
        datetime = dir0.split('/')[-1]
        C, top1_acc, top5_acc = None, None, None

        for dir1 in glob(dir0 + '/*'):
            # print(dir1)
            filename = dir1.split('/')[-1]
            if filename == 'config.yml':
                # print(dir1)
                config = yaml.unsafe_load(open(dir1))
                C = config.C

            if filename == 'result.yml':
                # print(dir1)
                result = yaml.unsafe_load(open(dir1))
                top1_acc = result['top1_acc'] * 100.0
                top5_acc = result['top5_acc'] * 100.0
                best_top1 = (top1_acc, C) if top1_acc > best_top1[0] else best_top1
                best_top5 = (top5_acc, C) if top5_acc > best_top5[0] else best_top5

                if datetime == '2021-06-30-12:32:32':
                    print(top1_acc)
                    print(top5_acc)

        # row = [lr, wd, epochs, hidden_dim, top1_acc, top5_acc, datetime]
        row = [C, top1_acc, top5_acc, datetime]
        data.append(row)
    # df = pd.DataFrame(data=data, columns=['lr', 'wd', 'epochs', 'hidden_dim', 'top1_acc', 'top5_acc', 'datetime'])
    # df = df.sort_values(by=['datetime'])
    df = pd.DataFrame(data=data, columns=['C', 'top1_acc', 'top5_acc', 'datetime'])
    df = df.sort_values(by=['C'])
    print(df)

    print()

    print(f'best_top1_acc: {best_top1[0]}, C: {best_top1[1]}')
    print(f'best_top5_acc: {best_top5[0]}, C: {best_top5[1]}')


if __name__ == '__main__':
    # read_color_mlp_tuning_log('./runs/tune/color_mlp')
    # read_color_mlp_tuning_log('./runs/tune/color_mlp/9')
    # read_color_mlp_tuning_log('./runs/tune/color_mlp/12')
    # read_mlp_tuning_log('./runs/tune/mlp/original_data')
    # read_color_mlp_tuning_log('./runs/tune/color_mlp/original_data')
    # read_color_mlp_tuning_log('./runs/tune/color_mlp/original_data', './color_mlp_log')
    # read_color_mlp_tuning_log('./runs/tune/color_mlp/new')
    read_svm_tuning_log('./runs_backup/tune/svm')
