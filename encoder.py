# from chitta


import torch.nn as nn
from torch.nn.init import orthogonal_, xavier_uniform_

""" Original 
5 encoder = Dense ( units =32 ,
6 activation =" relu ",
7 input_shape =( input_dim ,) ,
8 use_bias = True ,
9 kernel_regularizer =
10 OrthogonalWeights (
11 weightage =1. ,
12 axis =0) ,
13 kernel_constraint =
14 UnitNorm ( axis =0) ,
15 activity_regularizer =
16 SparseCovariance ( weightage =1.) ,
name =’encoder ’)"""

# Bing xlated
class OrthogonalWeights(nn.Module):
    def __init__(self, weightage, axis):
        super(OrthogonalWeights, self).__init__()
        self.weightage = weightage
        self.axis = axis

    def forward(self, w):
        if self.axis == 1:
            w = w.t()
        rows = w.shape[0]
        cols = w.shape[1]
        if rows > cols:
            flat_shape = (rows, rows // cols * cols)
        else:
            flat_shape = (cols // rows * rows, cols)
        a = torch.randn(flat_shape)
        u, _, v = torch.svd(a)
        q = u if rows > cols else v
        q = q.t().reshape(*w.shape)
        new_w = self.weightage * q[:rows, :cols] + (1 - self.weightage) * w
        return new_w


class UnitNorm(nn.Module):
    def __init__(self, axis=0):
        super(UnitNorm, self).__init__()
        self.axis = axis

    def forward(self, x):
        norms = x.norm(p=2, dim=self.axis, keepdim=True)
        return x / norms


class SparseCovariance(nn.Module):
    def __init__(self, weightage):
        super(SparseCovariance, self).__init__()
        self.weightage = weightage

    def forward(self, x):
        cov_matrix = torch.matmul(x.t(), x)
        sparse_cov_matrix = torch.mul(cov_matrix, torch.eye(cov_matrix.shape[0], cov_matrix.shape[1]))
        sparse_cov_matrix -= torch.mul(torch.eye(cov_matrix.shape[0], cov_matrix.shape[1]), cov_matrix)
        return self.weightage * sparse_cov_matrix


class PyTorchEncoder(nn.Module):
    def __init__(self, input_dim=10):
        super(PyTorchEncoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder_layer(x)


input_dim = 10
encoder = PyTorchEncoder(input_dim=input_dim)
