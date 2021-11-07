import numpy as np


def get_rgb_mean():
    single_channel = [i for i in range(0, 256)]
    single_channel = np.array(single_channel)
    mean = single_channel.mean()
    return [mean for _ in range(3)]


def get_rgb_std():
    single_channel = [i for i in range(0, 256)]
    single_channel = np.array(single_channel)
    std = single_channel.std()
    return [std for _ in range(3)]


def get_rgb_mean_and_std():
    single_channel = [i for i in range(0, 256)]
    single_channel = np.array(single_channel)
    mean = single_channel.mean()
    std = single_channel.std()
    return [mean for _ in range(3)], [std for _ in range(3)]


if __name__ == '__main__':
    # pixels = []
    # for r in range(0, 256):
    #     for g in range(0, 256):
    #         for b in range(0, 256):
    #             pixels.append([r, g, b])
    #
    # assert len(pixels) == 256 * 256 * 256
    #
    # pixels = np.array(pixels)
    # print(pixels.shape)
    #
    # mean = pixels.mean(axis=0).tolist()
    # std = pixels.std(axis=0).tolist()
    #
    # print(f'mean: {mean}')
    # print(f'std: {std}')

    mean, std = get_rgb_mean_and_std()

    print(mean)
    print(std)
