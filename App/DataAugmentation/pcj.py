import random

import numpy as np
from PIL import Image
from torchvision.transforms import transforms


class PartialColorJitter(object):

    def __init__(self, y_path, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2):
        super(PartialColorJitter, self).__init__()

        self.classes = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
        self.code = np.array([256 ** 2, 256 ** 1, 256 ** 0])
        self.color_jitter = transforms.ColorJitter(brightness=brightness,
                                                   contrast=contrast,
                                                   saturation=saturation,
                                                   hue=hue)
        self.y = np.load(y_path)

    def predict(self, img_arr: np.ndarray):
        """
        Predict each pixel's label in a given image array.
        :param img_arr: image array, dtype: uint8, value range: [0, 255]
        :return:
        """
        assert img_arr.dtype == np.uint8
        assert img_arr.shape[2] == 3
        img_arr = img_arr.reshape(-1, 3)
        img_indices = (img_arr * self.code).sum(1)
        return self.y[img_indices]

    def __call__(self, img: Image):
        img_arr = np.array(img)  # dtype: uint8, range: [0, 255]
        h, w, c = img_arr.shape
        # print(type(img))
        pred_y = self.predict(img_arr)
        # print(pred_Y.shape)
        indices = np.unique(pred_y)
        i = indices[random.randint(0, indices.shape[0] - 1)]
        mask = (pred_y == i).reshape(-1, 1).repeat(3, axis=1).reshape(h, w, c)
        roi = img_arr[mask].reshape(-1, 1, 3)
        img_arr[mask] = np.array(self.color_jitter(Image.fromarray(roi))).reshape(-1)
        return Image.fromarray(img_arr)


def get_mask(y, h, w, classes):
    color_dict = {'Black': [0, 0, 0],
                  'Blue': [0, 0, 255],
                  'Brown': [165, 42, 42],
                  'Gray': [128, 128, 128],
                  'Green': [0, 255, 0],
                  'Orange': [255, 165, 0],
                  'Pink': [255, 192, 203],
                  'Purple': [128, 0, 128],
                  'Red': [255, 0, 0],
                  'White': [255, 255, 255],
                  'Yellow': [255, 255, 0]}
    r, g, b = np.copy(y), np.copy(y), np.copy(y)
    for i, k in enumerate(classes):
        r_v, g_v, b_v = color_dict[k]
        r[r == i] = r_v
        g[g == i] = g_v
        b[b == i] = b_v

    return np.concatenate((np.expand_dims(r, 1),
                           np.expand_dims(g, 1),
                           np.expand_dims(b, 1)), 1).reshape(h, w, 3).astype(np.uint8)


if __name__ == '__main__':
    pcj = PartialColorJitter(y_path='../../CCN/data_backup/y.npy')
    img = Image.open('./n01532829_house_finch.jpeg')
    img_arr = np.array(img)
    h, w, c = img_arr.shape
    y = pcj.predict(img_arr)
    print(img_arr.shape)

    print(y.reshape(h, w))

    img_mask = get_mask(y, h, w, pcj.classes)
    print(img_mask.shape)

    import matplotlib.pyplot as plt

    plt.imshow(img_arr)
    plt.show()

    plt.imshow(img_mask)
    plt.show()

    # x


