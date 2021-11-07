import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    ref: https://github.com/Diego999/pyGAT/tree/b7b68866739fb5ae6b51ddb041e16f8cef07ba87
    """

    # def __init__(self, in_features, out_features, dropout, alpha, concat=True):
    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.dropout_p = dropout
        self.alpha = alpha
        # self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, h, adj):
        # Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # e = F.tanh(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
        # return F.tanh(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={str(self.in_features)}, out_features={str(self.out_features)}, alpha={str(self.alpha)})'


class ColorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, alpha, adjs):
        super(ColorMLP, self).__init__()
        self.adjs = adjs  # shape: [3, 11, 11] rgb channel adjacency matrix
        self.eye = torch.eye(output_dim).to(adjs.device)
        self.fc0 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        num_hidden_dim = hidden_dim // adjs.size(0)  # 6 = 18 // 3

        self.gals = [[GraphAttentionLayer(in_features=output_dim, out_features=num_hidden_dim, alpha=alpha)
                      for _ in range(num_heads)]
                     for _ in range(3)]  # r, b, g gals: list in list

        self.gals_names = ['rgal', 'ggal', 'bgal']

        for name, gals in zip(self.gals_names, self.gals):
            for i, gal in enumerate(gals):
                self.add_module('{}{}'.format(name, i), gal)

    def forward(self, x):
        # [m, hidden_dim] * [hidden_dim, output_dim] + [output_dim]
        # where [hidden_dim, output_dim] = [num_heads, hidden_dim, nhid] -> [hidden_dim, num_hidden_dim] -> [hidden_dim, output_dim]
        return F.tanh(self.fc0(x)).mm(torch.cat([torch.stack([gal(self.eye, self.adjs[i]) for gal in gals], dim=0)
                                                .mean(dim=0) for i, gals in enumerate(self.gals)],
                                                dim=1).t()) + self.bias

    def get_w(self):
        # shape: [3, output_dim, num_hidden_dim]
        return torch.stack([torch.stack([gal(self.eye, self.adjs[i]) for gal in gals], dim=0).mean(dim=0) for i, gals in
                            enumerate(self.gals)], dim=0)
