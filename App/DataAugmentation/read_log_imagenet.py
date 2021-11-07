from glob import glob

import torch

if __name__ == '__main__':
    root_dir = './runs/imagenet'
    root_dir = root_dir + '*' if root_dir == '/' else root_dir + '/*'
    print(root_dir)
    # print(glob(root_dir + '/*'))

    metric_is_acc_or_error = False

    for arch_dir in glob(root_dir):
        arch = arch_dir.split('/')[-1]
        # print(f'arch_dir: {arch_dir}')
        # print(f'arch: {arch}')
        for augment_dir in glob(arch_dir + '/*'):
            augment = augment_dir.split('/')[-1]
            # print(f'augment_dir: {augment_dir}')
            # print(f'augment: {augment}')
            for file_dir in glob(augment_dir + '/*'):
                filename = file_dir.split('/')[-1]
                # print(f'dir4: {dir4}')
                # print(f'filename: {filename}')
                # if filename == 'model_best.pth.tar':
                #     # print(f'dir4: {dir4}')
                # print(f'arch: {arch}, augment: {augment}, datetime: {datetime}, filename: {filename}')
                if filename == 'model_best.pth.tar':
                    # print(file_dir)
                    checkpoint = torch.load(file_dir, map_location='cpu')
                    best_acc1 = checkpoint['best_acc1']
                    acc5 = checkpoint['acc5']
                    epoch = checkpoint['epoch']

                    if metric_is_acc_or_error:
                        print('arch: {}, augment: {}, filename: {}, best_acc1: {}, acc5: {}, epoch: {}'.
                              format(arch, augment, filename, best_acc1, acc5, epoch))
                    else:
                        best_err1 = 100. - best_acc1
                        err5 = 100. - acc5
                        print('arch: {}, augment: {}, filename: {}, best_err1: {}, err5: {}, epoch: {}'.
                              format(arch, augment, filename, best_err1, err5, epoch))
