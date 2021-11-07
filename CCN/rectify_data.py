import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import config
from modules import ColorMLP


def group_tensor(X, y, classes, k):
    idx = classes.index(k)
    return X[y == idx], y[idx]


def load_X():
    X = []
    for (i, j, k) in itertools.product(range(256), range(256), range(256)):
        X.append([i, j, k])
    X = torch.tensor(X, dtype=torch.float)
    # print(X.size())
    return X


"""
Black: [0, 0, 0] - [43, 43, 43]
Gray: [44, 44, 44] - [244, 244, 244]
White: [245, 245, 245] - [255, 255, 255]
"""

bgw_dict = {'Black': (0, 43),
            'Gray': (44, 244),
            'White': (245, 255)}

"""
Black: [0, 0, 0] - [43, 43, 43]
Gray: [44, 44, 44] - [244, 244, 244]
White: [245, 245, 245] - [255, 255, 255]
"""


def get_bgw():
    white_start, white_end = 245, 255
    black_start, black_end = 0, 43
    gray_start, gray_end = 44, 244

    X = load_X()
    r, g, b = X.t()
    X_std = X.std(dim=1)
    sr_3 = math.sqrt(3.0)

    # white
    white_range_x = X[((white_start <= r) * (r <= white_end))
                      * ((white_start <= g) * (g <= white_end))
                      * ((white_start <= b) * (b <= white_end)), :]
    white_range_x_std = white_range_x.std(1)
    white_x = []
    white_min_offset = 1
    white_max_offset = white_range_x_std.max()
    interval = 1
    white_l = np.sqrt(np.power(white_range_x, 2).sum(1) - np.power(white_range_x_std, 2))

    white_ranges = np.arange(start=white_start, stop=(white_end + 1), step=interval).tolist()
    white_step = (white_max_offset - white_min_offset) / (len(white_ranges[:-1]) - 1)
    for i, b in enumerate(white_ranges[:-1]):
        start = b * sr_3
        end = white_end if len(white_ranges) - 1 == i + 1 else white_ranges[i + 1]
        end *= sr_3
        offset = white_min_offset + i * white_step
        roi = white_range_x[(start <= white_l) * (white_l <= end), :]
        white_x.append(roi[(roi.std(1) <= offset), :])

    # remove redundant
    white_x = torch.cat(white_x, dim=0).long()
    white_x_indices = (white_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long().numpy()
    unique_white_x_indices, white_x_indices_idx = np.unique(white_x_indices, return_index=True)
    white_x = (white_x[white_x_indices_idx]).float()

    # white_range_x_std = white_range_x.std(1)
    # white_x = []
    # white_l = np.sqrt(np.power(white_range_x, 2).sum(1) - np.power(white_range_x_std, 2))
    # white_stop = white_end if white_end % interval == 0 else (white_end // interval + 1) * interval
    # white_ranges = np.arange(start=white_start, stop=(white_stop + 1), step=interval).round(2).tolist()
    # white_step = (max_offset - min_offset) / (len(white_ranges[:-1]) - 1)
    # for i, b in enumerate(white_ranges[:-1]):
    #     start = b * sr_3
    #     end = white_end if i + 1 == len(white_ranges) - 1 else white_ranges[i + 1]
    #     end *= sr_3
    #     offset = min_offset + i * white_step
    #     roi = white_range_x[(start <= white_l) * (white_l <= end), :]
    #     white_x.append(roi[(roi.std(1) <= offset), :])
    #
    # white_x = torch.cat(white_x, dim=0).float()

    # black
    black_range_x = X[((black_start <= r) * (r <= black_end))
                      * ((black_start <= r) * (r <= black_end))
                      * ((black_start <= r) * (r <= black_end)), :]

    black_range_x_std = black_range_x.std(1)

    black_x = []
    black_min_offset = 1  # 2
    black_max_offset = 19  # 19
    sr_3 = math.sqrt(3.0)
    interval = 1

    black_l = np.sqrt(np.power(black_range_x, 2).sum(1) - np.power(black_range_x_std, 2))

    black_stop = black_end if black_end % interval == 0 else (black_end // interval + 1) * interval
    black_ranges = np.arange(start=black_start, stop=(black_stop + 1), step=interval).tolist()
    black_step = (black_max_offset - black_min_offset) / (len(black_ranges[:-1]) - 1)
    for i, b in enumerate(black_ranges[:-1]):
        start = b * sr_3
        end = black_end if len(black_ranges) - 1 == i + 1 else black_ranges[i + 1]
        end *= sr_3
        offset = black_max_offset - i * black_step
        roi = black_range_x[(start <= black_l) * (black_l <= end), :]
        black_x.append(roi[(roi.std(1) <= offset), :])

    # black_x = torch.cat(black_x, dim=0).float()
    # remove redundant
    black_x = torch.cat(black_x, dim=0).long()
    black_x_indices = (black_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long().numpy()
    unique_black_x_indices, black_x_indices_idx = np.unique(black_x_indices, return_index=True)
    black_x = (black_x[black_x_indices_idx]).float()

    # gray = bgw - black - white
    gray_offset = 12
    bgw_x = X[X_std <= gray_offset, :]
    bgw_indices = (bgw_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    white_indices = (white_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    assert white_indices.numpy().shape[0] == np.unique(white_indices.numpy()).shape[0]
    black_indices = (black_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    # print(black_indices.numpy().shape[0] - np.unique(black_indices.numpy()).shape[0])
    assert black_indices.numpy().shape[0] == np.unique(black_indices.numpy()).shape[0]

    gray_mask = (~(np.in1d(bgw_indices.numpy(), white_indices.numpy()))) * \
                (~(np.in1d(bgw_indices.numpy(), black_indices.numpy())))
    gray_x = bgw_x[gray_mask, :]

    return black_x, gray_x, white_x


def rectify(X, net, mean, std, classes):
    X_indices = (X * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)

    # handle 'Black', 'Gray', 'White'
    black_x, gray_x, white_x = get_bgw()
    # print(black_x.shape)
    black_x_indices = (black_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)
    gray_x_indices = (gray_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)
    white_x_indices = (white_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)

    # NOTE: use -1 as default init value
    y = - torch.ones_like(X_indices).long()
    black_mask = np.in1d(X_indices.numpy(), black_x_indices.numpy())
    print(f'black_mask: {black_mask.sum()}')
    y[black_mask] = classes.index('Black')

    print('Black num: {0}'.format((y == classes.index('Black')).sum().item()))

    gray_mask = np.in1d(X_indices.numpy(), gray_x_indices.numpy())
    print(f'gray_mask: {gray_mask.sum()}')
    y[gray_mask] = classes.index('Gray')

    print('Gray num: {0}'.format((y == classes.index('Gray')).sum().item()))

    white_mask = np.in1d(X_indices.numpy(), white_x_indices.numpy())
    print(f'white_mask: {white_mask.sum()}')
    y[white_mask] = classes.index('White')

    print('White num: {0}'.format((y == classes.index('White')).sum().item()))

    # handle others
    others_X = X[((~black_mask) * (~gray_mask) * (~white_mask))]
    print(others_X.shape)
    others_logits = net((others_X - mean) / std)
    others_prob = F.softmax(others_logits, dim=1)
    bgw_classes_indices = list({classes.index('Black'), classes.index('Gray'), classes.index('White')})
    others_prob[:, bgw_classes_indices] = -1.0
    rectified_y = others_prob.argmax(dim=1)
    # back to the y
    others_indices = (others_X * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)
    others_mask = np.in1d(X_indices.numpy(), others_indices.numpy())
    y[others_mask] = rectified_y

    return y


def print_cleaned_data_bgw_info():
    classes = config['classes']
    # original
    data = np.load('./data_backup/cleaned_data.npy')
    X = data[:, 0:3]
    y = data[:, 3]

    for k in ['Black', 'Gray', 'White']:
        x, _ = group_tensor(X, y, classes, k)
        x_std = x.std(axis=1)
        print(f'{k}_x_std: min: {x_std.min()}, max: {x_std.max()}, mean: {x_std.mean()}')


def print_aug_data_bgw_info():
    classes = config['classes']
    # original
    data = np.load('./data_backup/aug_data.npy')
    X = data[:, 0:3]
    y = data[:, 3]

    for k in ['Black', 'Gray', 'White']:
        x, _ = group_tensor(X, y, classes, k)
        x_std = x.std(axis=1)
        print(f'{k}_x_std: min: {x_std.min()}, max: {x_std.max()}, mean: {x_std.mean()}')


def print_color_mlp_predicted_bgw_info(color_mlp):
    classes = config['classes']
    mean = torch.tensor(config['mean'])
    std = torch.tensor(config['std'])

    # original
    X = load_X()
    y = F.softmax(color_mlp((X - mean) / std), dim=1).argmax(dim=1)

    for k in ['Black', 'Gray', 'White']:
        x, _ = group_tensor(X, y, classes, k)
        if x.shape[0] == 0:
            continue
        x_std = x.std(axis=1)
        print(f'{k}_x_std: min: {x_std.min()}, max: {x_std.max()}, mean: {x_std.mean()}')


def visualize_data(data, elev=30, azim=45, c=None, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d(left=0, right=255)
    ax.set_xlabel("R")
    ax.set_ylim3d(bottom=0, top=255)
    ax.set_ylabel("G")
    ax.set_zlim3d(bottom=0, top=255)
    ax.set_zlabel("B")
    pixel_colors = (data / 255.).tolist()

    ax.view_init(elev=elev, azim=azim)

    r, g, b = data.T

    if c:
        ax.scatter(xs=r, ys=g, zs=b, facecolor=c, marker=".")
    else:
        ax.scatter(xs=r, ys=g, zs=b, facecolor=pixel_colors, marker=".")
    plt.show()


if __name__ == '__main__':
    X = load_X()
    print(X.shape)
    classes = config['classes']
    mean = torch.tensor(config['mean'])
    std = torch.tensor(config['std'])

    # load network
    color_mlp_path = './runs_backup/train/color_mlp/2021-06-28-16:51:56/last.pth.tar'
    color_mlp = ColorMLP(3, 18, 11, 5, 0.2, config['rgb_channel_adj'])
    color_mlp.load_state_dict(torch.load(color_mlp_path, map_location=torch.device('cpu')))

    y = rectify(X, color_mlp, mean, std, classes)

    print(X.shape, y.shape)

    np.save(f'./data_backup/y', y.numpy())
    print('saved!')

    # black_x, gray_x, white_x = get_bgw()
    #
    # print(black_x.shape, gray_x.shape, white_x.shape)
    # # print(white_x.std(1))
    # gray_indices = (gray_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    # white_indices = (white_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    # black_indices = (black_x * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1).long()
    #
    # gray_white_mask = np.in1d(gray_indices.numpy(), white_indices.numpy())
    # gray_black_mask = np.in1d(gray_indices.numpy(), black_indices.numpy())
    #
    # print(gray_white_mask.sum())
    # print(gray_black_mask.sum())

    # x = torch.cat((black_x, gray_x), 0)
    #
    # visualize_data(x.numpy(), 30, 45)
    # visualize_data(x.numpy(), 30, 90)
    # visualize_data(x.numpy(), 30, 135)

    #
    # visualize_data(black_x.numpy(), 30, 45)
    # visualize_data(black_x.numpy(), 30, 90)
    # visualize_data(black_x.numpy(), 30, 135)
    #
    # visualize_data(gray_x.numpy(), 30, 45)
    # visualize_data(gray_x.numpy(), 30, 90)
    # visualize_data(gray_x.numpy(), 30, 135)
    # #
    # visualize_data(white_x.numpy(), 30, 45)
    # visualize_data(white_x.numpy(), 30, 90)
    # visualize_data(white_x.numpy(), 30, 135)

    # visualize_data(white_x.numpy(), 30, 45, 'g')
    # visualize_data(white_x.numpy(), 30, 90, 'g')
    # visualize_data(white_x.numpy(), 30, 135, 'g')

    # def get_white(offset):
    #     white = [(i, j, k) for (i, j, k) in itertools.product(range(245, 256), range(245, 256), range(245, 256))]
    #     white = torch.tensor(white).float()
    #     return white[white.std(1) <= offset, :]
    #
    # white_x = get_white(7)
    # print(white_x.shape)
    #
    # visualize_data(white_x.numpy(), 30, 45)
    # visualize_data(white_x.numpy(), 30, 90)
    # visualize_data(white_x.numpy(), 30, 135)

    # print(f'cleaned_data')
    # print_cleaned_data_bgw_info()
    #
    # print(f'aug_data')
    # print_aug_data_bgw_info()
    #
    # print('color_mlp predicted data')
    # # load network
    # color_mlp_path = './runs_backup/train/color_mlp/2021-06-28-16:51:56/last.pth.tar'
    # color_mlp = ColorMLP(3, 18, 11, 5, 0.2, config['rgb_channel_adj'])
    # color_mlp.load_state_dict(torch.load(color_mlp_path, map_location=torch.device('cpu')))
    # print_color_mlp_predicted_bgw_info(color_mlp)
    #
    # offset = 7
    #
    # _bgw = get_bgw(offset)
    # print(_bgw.shape)
    # visualize_data(_bgw, 30, 45)
    # visualize_data(_bgw, 30, 90)
    # visualize_data(_bgw, 30, 135)

# X = load_X()
# print(X.shape)
# classes = config['classes']
# mean = torch.tensor(config['mean'])
# std = torch.tensor(config['std'])
#
# # load network
# color_mlp_path = './runs_backup/train/color_mlp/2021-06-28-16:51:56/last.pth.tar'
# color_mlp = ColorMLP(3, 18, 11, 5, 0.2, config['rgb_channel_adj'])
# color_mlp.load_state_dict(torch.load(color_mlp_path, map_location=torch.device('cpu')))
#
# y = rectify(X, color_mlp, mean, std, classes)
#
# print(X.shape, y.shape)

# black_x, black_y = group_tensor(X, y, classes, 'Black')
# print(black_x.shape)
#
# visualize_data(black_x, 30, 45)
# visualize_data(black_x, 30, 90)
# visualize_data(black_x, 30, 135)

# np.save(f'./data_backup/y', y.numpy())

# classes = config['classes']
# # print(classes.index('Black'))
# # print(torch.tensor(classes.index('Black')))
# black_x, black_y = bgw(bgw_dict['Black'], 'Black', classes)
# print(black_x.shape)
# # visualize_data(black_x)
# # visualize_data(black_x, 30, 135)
#
# visualize_data(black_x, 30, 45)
# visualize_data(black_x, 30, 90)
# visualize_data(black_x, 30, 135)
#
# gray_x, gray_y = bgw(bgw_dict['Gray'], 'Gray', classes)
# print(gray_x.shape)
# # visualize_data(gray_x, 30, 135)
#
# white_x, white_y = bgw(bgw_dict['White'], 'White', classes)
# print(white_x.shape)

# visualize_data(white_x, 30, 135)

# visualize_data(torch.cat((black_x, gray_x, white_x), 0), 30, 135)

# import numpy as np
#
# classes = config['classes']
# # original
# data = np.load('./data_backup/cleaned_data.npy')
# X = data[:, 0:3]
# y = data[:, 3]
#
# # print
# black_x, _ = group_tensor(X, y, classes, 'Black')
# black_x_std = black_x.std(axis=1)
#
# print(black_x.max())
# print(f'black_x_std: {black_x_std}')
# print(f'min: {black_x_std.min()}, max: {black_x_std.max()}, mean: {black_x_std.mean()}')
#
# gray_x, _ = group_tensor(X, y, classes, 'Gray')
# gray_x_std = gray_x.std(axis=1)
#
# print(gray_x.max())
# print(f'gray_x_std: {gray_x_std}')
# print(f'min: {gray_x_std.min()}, max: {gray_x_std.max()}, mean: {gray_x_std.mean()}')
#
# white_x, _ = group_tensor(X, y, classes, 'White')
# white_x_std = white_x.std(axis=1)
#
# print(white_x.max())
# print(f'white_x_std: {white_x_std}')
# print(f'min: {white_x_std.min()}, max: {white_x_std.max()}')
#
# classes = config['classes']
#
# X = load_X()
# indices = (X * torch.tensor([256 ** 2, 256 ** 1, 256 ** 0])).sum(dim=1)
# print(indices[:10])
# # normalize data
# mean = torch.tensor(config['mean'])
# std = torch.tensor(config['std'])
# X = (X - mean) / std
#
# # load network
# color_mlp_path = './runs_backup/train/color_mlp/2021-06-28-16:51:56/last.pth.tar'
# color_mlp = ColorMLP(3, 18, 11, 5, 0.2, config['rgb_channel_adj'])
# color_mlp.load_state_dict(torch.load(color_mlp_path, map_location=torch.device('cpu')))
#
# # predict
# y = (F.softmax(color_mlp(X), dim=1)).argmax(dim=1)
#
# # print
# black_x, _ = group_tensor(X, y, classes, 'Black')
# black_x = black_x * std + mean
# black_x_std = black_x.std(dim=1)
#
# print(black_x.max())
# print(f'black_x_std: {black_x_std}')
# print(f'min: {black_x_std.min()}, max: {black_x_std.max()}, mean: {black_x_std.mean()}')
#
# gray_x, _ = group_tensor(X, y, classes, 'Gray')
# gray_x = gray_x * std + mean
# gray_x_std = gray_x.std(dim=1)
#
# print(gray_x.max())
# print(f'gray_x_std: {gray_x_std}')
# print(f'min: {gray_x_std.min()}, max: {gray_x_std.max()}, mean: {gray_x_std.mean()}')
#
# # white_x, _ = group_tensor(X, y, classes, 'White')
# # white_x = white_x * std + mean
# # white_x_std = white_x.std(dim=1)
# #
# # print(white_x.max())
# # print(f'white_x_std: {white_x_std}')
# # print(f'min: {white_x_std.min()}, max: {white_x_std.max()}')
#
# # rectify
#
# new_X = load_rectified_X(color_mlp, mean, std, classes)
#
# print(new_X.shape)
