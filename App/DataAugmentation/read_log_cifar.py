import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Training Logs on CIFAR-10/100')

    parser.add_argument('--num_epochs', default=10, type=int,
                        help='scalers from last (num) epochs')

    parser.add_argument('--log_dir', default='./runs', type=str,
                        help='log dir, default: ./runs')

    parser.add_argument('--metric', type=str, choices=['acc', 'err'], default='err')

    parser.add_argument('--output_dir', default='./', type=str,
                        help='where output file to be saved, default: ./')

    args = parser.parse_args()

    # root_dir = os.path.join(args.log_dir, '/*/*/*/*')
    root_dir = args.log_dir + '*' if args.log_dir[-1] == '/' else args.log_dir + '/*'
    print(args.log_dir)
    print(root_dir)
    # print(glob(root_dir + '/*'))

    for dataset_dir in glob(root_dir):
        log_data = []  # each log
        avg_data = []  # mean and std of one network with one augment
        dataset = dataset_dir.split('/')[-1]
        print(f'dataset_dir: {dataset_dir}')
        print(f'dataset_name: {dataset}')
        for arch_dir in glob(dataset_dir + '/*'):
            arch = arch_dir.split('/')[-1]
            print(f'arch_dir: {arch_dir}')
            print(f'arch: {arch}')
            for augment_dir in glob(arch_dir + '/*'):
                augment = augment_dir.split('/')[-1]
                # print(f'augment_dir: {augment_dir}')
                # print(f'augment: {augment}')

                # for avg_data
                top1 = []
                top5 = []

                for datetime_dir in glob(augment_dir + '/*'):
                    datetime = datetime_dir.split('/')[-1]
                    print(f'datetime_dir: {datetime_dir}')
                    print(f'datetime: {datetime}')

                    event_acc = EventAccumulator(datetime_dir)
                    event_acc.Reload()
                    # print(event_acc.Tags())
                    # test_top5_acc = [(s.step, s.value) for s in event_acc.Scalars('test/top5_acc')]
                    test_top1_acc = [s.value for s in event_acc.Scalars('test/top1_acc')][-args.num_epochs:]
                    test_top5_acc = [s.value for s in event_acc.Scalars('test/top5_acc')][-args.num_epochs:]

                    # mean_test_top1_acc = np.array(test_top1_acc).mean()
                    # mean_test_top5_acc = np.array(test_top5_acc).mean()

                    # print(test_top1_acc)
                    # print(test_top5_acc)
                    if args.metric == 'acc':
                        mean_test_top1_acc = np.array(test_top1_acc).mean()
                        mean_test_top5_acc = np.array(test_top5_acc).mean()
                        top1.append(mean_test_top1_acc)
                        top5.append(mean_test_top5_acc)
                        log_data.append([arch,
                                         augment,
                                         np.round(mean_test_top1_acc, 2),
                                         np.round(mean_test_top5_acc, 2),
                                         test_top1_acc,
                                         test_top5_acc,
                                         datetime])
                    else:
                        test_top1_err = 100. - np.array(test_top1_acc)
                        test_top5_err = 100. - np.array(test_top5_acc)
                        mean_test_top1_err = test_top1_err.mean()
                        mean_test_top5_err = test_top5_err.mean()
                        top1.append(mean_test_top1_err)
                        top5.append(mean_test_top5_err)
                        log_data.append([arch,
                                         augment,
                                         np.round(mean_test_top1_err, 2),
                                         np.round(mean_test_top5_err, 2),
                                         test_top1_err,
                                         test_top5_err,
                                         datetime])



                    # for file_dir in glob(datetime_dir + '/*'):
                    #     filename = file_dir.split('/')[-1]
                    #     # print(f'dir4: {dir4}')
                    #     # print(f'filename: {filename}')
                    #     if filename == 'result.yml':
                    #         # print(f'dir4: {dir4}')

                top1 = np.asarray(top1)
                top5 = np.asarray(top5)
                avg_data.append([arch,
                                 augment,
                                 '{0} ± {1}'.format(np.round(top1.mean(), 2).item(),
                                                    np.round(top1.std(), 2).item()),
                                 '{0} ± {1}'.format(np.round(top5.mean(), 2).item(),
                                                    np.round(top5.std(), 2).item()),
                                 top1.shape[0]])

        if args.metric == 'acc':
            log_df = pd.DataFrame(data=log_data,
                                  columns=['arch',
                                           'augment',
                                           'mean_top1_acc',
                                           'mean_top5_acc',
                                           'top1_acc_from_last_{0}_epochs'.format(args.num_epochs),
                                           'top5_acc_from_last_{0}_epochs'.format(args.num_epochs),
                                           'datetime'])

        # df.sort_values(by=['backbone', 'augment'])
        # custom_dict = {'none': 0, 'cj': 1, 'pcj': 2, 'rc_rhf_none': 3, 'rc_rhf_cj': 4, 'rc_rhf_pcj': 5}
        # df = df.sort_values(by=['backbone', 'augment'], key=lambda x: x.map(custom_dict))

            augment_order = ['none', 'cj', 'pcj', 'rc_rhf_none', 'rc_rhf_cj', 'rc_rhf_pcj']
            augment_dtype = pd.CategoricalDtype(augment_order, ordered=True)
            log_df['augment'] = log_df['augment'].astype(augment_dtype)

            log_df = log_df.sort_values(by=['arch', 'augment'])
            # df = df.sort_values(by=['augment'])
            # df_dict[dataset] = df
            # log_df.to_csv(f'{args.log_dir}/{dataset}.csv')
            log_df.to_csv(os.path.join(args.output_dir, dataset + '_acc.csv'))

            avg_df = pd.DataFrame(data=avg_data,
                                  columns=['arch',
                                           'augment',
                                           'top1_acc',
                                           'top5_acc',
                                           'times'])
            avg_df['augment'] = avg_df['augment'].astype(augment_dtype)
            avg_df = avg_df.sort_values(by=['arch', 'augment'])
            # avg_df.to_csv(f'{args.log_dir}/{dataset}_avg.csv')
            avg_df.to_csv(os.path.join(args.output_dir, dataset + '_acc_avg.csv'))

        else:
            log_df = pd.DataFrame(data=log_data,
                                  columns=['arch',
                                           'augment',
                                           'mean_top1_err',
                                           'mean_top5_err',
                                           'top1_err_from_last_{0}_epochs'.format(args.num_epochs),
                                           'top5_err_from_last_{0}_epochs'.format(args.num_epochs),
                                           'datetime'])

            # df.sort_values(by=['backbone', 'augment'])
            # custom_dict = {'none': 0, 'cj': 1, 'pcj': 2, 'rc_rhf_none': 3, 'rc_rhf_cj': 4, 'rc_rhf_pcj': 5}
            # df = df.sort_values(by=['backbone', 'augment'], key=lambda x: x.map(custom_dict))

            augment_order = ['none', 'cj', 'pcj', 'rc_rhf_none', 'rc_rhf_cj', 'rc_rhf_pcj']
            augment_dtype = pd.CategoricalDtype(augment_order, ordered=True)
            log_df['augment'] = log_df['augment'].astype(augment_dtype)

            log_df = log_df.sort_values(by=['arch', 'augment'])
            # df = df.sort_values(by=['augment'])
            # df_dict[dataset] = df
            # log_df.to_csv(f'{args.log_dir}/{dataset}.csv')
            log_df.to_csv(os.path.join(args.output_dir, dataset + '_err.csv'))

            avg_df = pd.DataFrame(data=avg_data,
                                  columns=['arch',
                                           'augment',
                                           'top1_err',
                                           'top5_err',
                                           'times'])
            avg_df['augment'] = avg_df['augment'].astype(augment_dtype)
            avg_df = avg_df.sort_values(by=['arch', 'augment'])
            # avg_df.to_csv(f'{args.log_dir}/{dataset}_avg.csv')
            avg_df.to_csv(os.path.join(args.output_dir, dataset + '_err_avg.csv'))
