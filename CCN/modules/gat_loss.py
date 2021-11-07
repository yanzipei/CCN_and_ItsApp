import torch
import torch.nn as nn


class GATLoss(nn.Module):
    """
    ref: https://arxiv.org/pdf/1611.07308.pdf page:2 formula:4
    """

    def __init__(self, l, adjs):
        super(GATLoss, self).__init__()
        assert l.size(1) == adjs.size(0)
        self.l = l  # shape: [1, 3]
        self.adjs = adjs  # [3, 11, 11]
        # self.eyes = torch.stack([torch.eye(adjs.size(1)) for _ in range(adjs.size(0))], dim=0).to(adjs.device)
        # assert self.adjs.size() == self.eyes.size()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, w):
        """
        :param w: shape: [3, 11, 6]
        :return: a tensor
        """
        return torch.einsum('ab, bcd -> acd', self.l, self.criterion(w.bmm(w.transpose(2, 1)), self.adjs)).mean()
