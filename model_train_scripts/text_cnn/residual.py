# from https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
import torch.nn as nn
import torch.nn.functional as F


# class Highway(nn.Module):
#     def __init__(self, size, num_layers, f):

#         super(Highway, self).__init__()

#         self.num_layers = num_layers

#         self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

#         self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

#         self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

#         self.f = f

#     def forward(self, x):
#         """
#             :param x: tensor with shape of [batch_size, size]
#             :return: tensor with shape of [batch_size, size]
#             applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
#             f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
#             and ⨀ is element-wise multiplication
#             """

#         for layer in range(self.num_layers):
#             gate = F.sigmoid(self.gate[layer](x))

#             nonlinear = self.f(self.nonlinear[layer](x))
#             linear = self.linear[layer](x)

#             x = gate * nonlinear + (1 - gate) * linear

#         return x

class ResidualMLP(nn.Module):
    def __init__(self, size ,f):
        super(ResidualMLP, self).__init__()
        self.nonlinear = nn.Linear(size, size)
        self.batchnorm = nn.BatchNorm1d(size)
        self.f=f    
    def forward(self, x):
        #nonlinear layer
        out = self.f(self.nonlinear(x))
        out=self.f(out)
        #batchnorm
        out = self.batchnorm(out)
        #nonlinearity
        out = self.f(out)
        #addition
        out = x + out
        return out
# From https://gist.github.com/a-rodin/d4f2ab5d7eb9d9887b26f28144e4ffdf
# class RHN (nn.Module):
    
#     def __init__(self, in_features, out_features, num_layers):
#         super().__init__()
#         self.transform = []
#         self.gate = []
#         for i in range(num_layers):
#             transform = Bilinear(in_features, out_features)
#             gate = Bilinear(in_features, out_features)
#             setattr(self, 'transform%d' % i, transform)
#             setattr(self, 'gate%d' % i, gate)
#             self.transform.append(transform)
#             self.gate.append(gate)
        
#     def forward(self, seq, h):
#         for x in seq:
#             for transform, gate in zip(self.transform, self.gate):
#                 gate_value = F.sigmoid(gate(x, h))
#                 h = F.tanh(transform(x, h)) * gate_value + h * (1 - gate_value)
#         return h