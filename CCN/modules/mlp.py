# from mish import Mish


"""
fork from: https://github.com/digantamisra98/Mish/blob/master/Mish/Torch
"""

import torch
import torch.nn.functional as F
from torch import nn



@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        # super().__init__()
        super(Mish, self).__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        # self.act = Mish() #nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)

    def forward(self, x):
        # return self.fc1(self.act(self.fc0(x)))
        return self.fc1(self.tanh(self.fc0(x)))


if __name__ == '__main__':
    # test
    mlp = MLP(3, 18, 11)
    print(mlp)
    print(mlp(torch.randn([3, 3])))
