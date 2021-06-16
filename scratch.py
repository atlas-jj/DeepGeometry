import numpy as np
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import abc


class LinearExpandingLayer(nn.Module):
    """
    linear expanding layer to learn weights for each basis
    return the same dim as the input but with weight ratios
    """

    def __init__(self, _basis_vector_dim, _basis_vector_num):
        super(LinearExpandingLayer, self).__init__()
        self.basis_vector_dim = _basis_vector_dim
        self.weight = Parameter(torch.Tensor(_basis_vector_num))
        self.register_parameter('bias', None)  # without bias term
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        repeated_weight = self.weight.repeat_interleave(self.basis_vector_dim)
        return x * repeated_weight

layer = LinearExpandingLayer(512, 7)
# x = torch.randn(10, 512*7)
x = torch.cat([torch.zeros(10,512), torch.ones(10,512), torch.ones(10,512)*2, torch.ones(10,512)*3, torch.ones(10,512)*4, torch.ones(10,512)*5, 6*torch.ones(10,512)],1)
y = layer(x)
out = torch.sum(y)
out.backward()
params = next(layer.parameters())
print(x)
print(params.grad)  # will be 4 times x because of the repeated weights. Autograd has taken the repetition into account


