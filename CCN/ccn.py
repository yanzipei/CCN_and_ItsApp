import numpy as np

from CCN.config import config


class CCN(object):
    def __init__(self, y_path):
        super(CCN, self).__init__()
        self.y = np.load(y_path)
        self.code = np.array([256 ** 2, 256 ** 1, 256 ** 0])
        self.classes = config['classes']
        assert self.y.shape[0] == 256 ** 3

    def __call__(self, img_arr: np.ndarray):
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


if __name__ == '__main__':
    ccn = CCN(y_path='./data/y.npy')
    img = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    print(img.shape)
    pred_y = ccn(img)
    print(pred_y)
