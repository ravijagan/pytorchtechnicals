
from torch import nn
from torch.nn.utils.parametrizations import orthogonal  as ortho
import torch.nn.functional as F
import torch
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

import torch


# Use torch.nn.Module to create models
class AutoEncoder(torch.nn.Module):
    def __init__(self, features: int, hidden_size: int):
        # Necessary in order to log C++ API usage and other internals
        super().__init__()
        #self.h = int(hidden_size/2)
        self.lin1 = nn.Linear(features, hidden_size)
        #self.lin2 = nn.Linear(hidden_size, self.h)
        self.relu1 = nn.ReLU(hidden_size)
        self.encoder = self.relu1 #(self.lin1)# self.relu1
        self.decoder1 = torch.nn.Linear(hidden_size, features)

    def forward(self, X):
        out1 = self.lin1(X)
        out2 = self.relu1(out1)
        # what we want self.decoder1(self.relu1(self.lin1(X)))
        #self.enc = ortho(out)  # but the axis ?
        self.dec = self.decoder1(out2)
        self.decoder = self.dec #
        return self.decoder

    def encode(self, X):
        return self.encoder(X)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, op_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.siz2  = int(hidden_size/2)
        self.l2 = nn.Linear(hidden_size, self.siz2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(self.siz2, op_size) # op_size 2 for binary


    def forward(self, x):
        out = self.l1(x) # self.l1 = nn.Linear(input_size, hidden_size)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        out = torch.clamp(out, 0.0, 1.0)
        return out


class logisticModel(nn.Module):
    def __init__(self, n_input_features):
        super(logisticModel, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
