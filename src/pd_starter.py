import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import _pickle as cPickle

import os, sys, pdb, logging
import math, random
from copy import deepcopy
from numpy.random import multinomial

from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.optimize import fmin

import torch
import torch.nn as nn
import torch.autograd
from torch.utils import data
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from adamw import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, classification_report, r2_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Make experiments deterministic for comparability
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=8)

# Output Layer for FCN architecture with an optional activation
class out_layer(nn.Module):
    def __init__(self, in_channels, out_channels = 1, gap_size = 1200, cl = False):
        super(out_layer, self).__init__()
        self.in_channels = in_channels
        self.gap = nn.AvgPool1d(kernel_size = gap_size, padding = 1)
        self.lin = nn.Linear(in_channels, out_channels)
        #nn.init.xavier_uniform_(self.lin.weight.data, gain=5.0/3.0)
        self.activation = nn.Sigmoid() if cl else lambda x: x
    def forward(self, x):
        gap = self.gap(x).view(-1, self.in_channels)
        return self.activation(self.lin(gap))

# Basic convolutional block of Conv, BN, Dropout, Activation
class base_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, elu=False, do = 0.0, ks = 1, **kwargs):
        super(base_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias = False, kernel_size = ks, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dp = nn.Dropout(p = do) if do > 0.0 else lambda x: x
        self.activation = nn.ELU() if elu else nn.RReLU()
        self.conv.weight.data.normal_(0, 0.01)
        #nn.init.kaiming_uniform_(self.conv.weight.data)
    def forward(self, x):
        return self.activation(self.dp(self.bn(self.conv(x))))
'''
# Fast inception block with residual connections and dilation
class incept_block(nn.Module):
    def __init__(self, in_channels, dilation=False):
        super(incept_block, self).__init__()
        self.branch_1 = base_conv1d(in_channels, 64, ks=1)
        self.branch_2 = base_conv1d(64, 64, ks=3, padding=1)
        if dilation:
            self.branch_3 = base_conv1d(64, 64, ks=3, padding=2, dilation=2)
            self.branch_4 = base_conv1d(64, 64, ks=3, padding=4, dilation=4)
        else:
            self.branch_3 = base_conv1d(64, 64, ks=3, padding=1)
            self.branch_4 = base_conv1d(64, 64, ks=3, padding=1)

        #self.mpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(branch1)
        branch3 = self.branch_3(branch2)
        branch4 = self.branch_4(branch3)
        #branch1 = self.branch_1(self.mpool(x))
        return torch.cat([branch1, branch2, branch3, branch4], 1)
'''

class incept_block(nn.Module):
    def __init__(self, in_channels, dilation=False):
        super(incept_block, self).__init__()
        self.branch_1 = base_conv1d(in_channels, 64, ks=1)
        if dilation:
            self.branch_2 = base_conv1d(64, 64, ks=3, padding=2, dilation=2)
            self.branch_3 = base_conv1d(64, 64, ks=3, padding=4, dilation=4)
            self.branch_4 = base_conv1d(64, 64, ks=3, padding=8, dilation=8)
            self.branch_5 = base_conv1d(64, 64, ks=3, padding=16, dilation=16)
        else:
            self.branch_2 = base_conv1d(64, 64, ks=3, padding=1)
            self.branch_3 = base_conv1d(64, 64, ks=3, padding=1)
            self.branch_4 = base_conv1d(64, 64, ks=3, padding=1)
            self.branch_5 = base_conv1d(64, 64, ks=3, padding=1)

        #self.mpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        branch2 = self.branch_2(self.branch_1(x))
        branch3 = self.branch_3(branch2)
        branch4 = self.branch_4(branch3)
        branch5 = self.branch_5(branch4)
        #branch1 = self.branch_1(self.mpool(x))
        return torch.cat([branch5, branch2, branch3, branch4], 1)



# FCN Architecture with optional Squeeze and Excite Layers
class Arch0(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, gap_size = 1200, cl = False, noise = 0.0):
        super(Arch0, self).__init__()   
        self.noise = noise
        self.conv1 = base_conv1d(in_channels = in_channels, out_channels = 128, elu = True, ks = 7, padding = 3)
        self.conv2 = base_conv1d(in_channels = 128, out_channels = 256, elu = True, ks = 5, padding = 2)
        # Can have a loop to add several inception blocks w/o dilation if wanted
        self.conv3 = base_conv1d(in_channels = 256, out_channels = 128, elu = True, ks = 3)
        self.out = out_layer(in_channels = 128, out_channels = out_channels, gap_size = gap_size, cl = cl)
    def forward(self, x):
        if self.training and self.noise != 0.0:
            x = x + Variable(x.data.new(x.size()).normal_(0.0, self.noise))
        return self.out(self.conv3(self.conv2(self.conv1(x))))

# FCN Architecture where the last layer is dilated Inception
class Arch1(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, gap_size = 1200, cl = False, noise = 0.0):
        super(Arch1, self).__init__()
        self.noise = noise
        self.conv1 = base_conv1d(in_channels = in_channels, out_channels = 128, ks = 7, padding = 3)
        self.conv2 = base_conv1d(in_channels = 128, out_channels = 256, ks = 5, padding = 2)
        # Can have a loop to add several inception blocks w/o dilation if wanted
        self.conv3 = incept_block(in_channels = 256, dilation = True)
        self.out = out_layer(in_channels = 256, out_channels= out_channels, gap_size = gap_size, cl = cl)
    def forward(self, x):
        if self.training and self.noise != 0.0:
            x = x + Variable(x.data.new(x.size()).normal_(0.0, self.noise))
        return self.out(self.conv3(self.conv2(self.conv1(x))))

# FCN Architecture with Inception Layers instead of Basic Conv
class Arch2(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, gap_size = 1200, cl = False, noise = 0.0):
        super(Arch2, self).__init__()
        self.noise = noise
        self.conv1 = incept_block(in_channels = in_channels)
        self.conv2 = incept_block(in_channels = 256, dilation=True)
        # Can have a loop to add several inception blocks w/o dilation if wanted
        self.conv3 = incept_block(in_channels = 256, dilation=True)
        self.out = out_layer(in_channels = 256, out_channels = out_channels, gap_size = gap_size, cl = cl)
    def forward(self, x):
        if self.training and self.noise != 0.0:
            x = x + Variable(x.data.new(x.size()).normal_(0.0, self.noise))
        return self.out(self.conv3(self.conv2(self.conv1(x))))

class sensor_dataset(data.Dataset):
    def __init__(self, list_IDs, data_set, meta_dict, cols = ['accnorm', 'gyronorm'], encoding = 'num_label'):
        self.list_IDs = list_IDs
        self.data_set = data_set
        self.meta_dict = meta_dict
        self.encoding = encoding
        self.cols = cols
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        x = np.asarray(self.data_set[ID-1][self.cols].values)
        X = torch.Tensor(x.transpose()).float()
        y = np.asarray(self.meta_dict[self.meta_dict['idx'] == ID][self.encoding])
        return X, y

# Loss implemented in PyTorch for training purpose
def relaxed_l2_loss(y, yhat, cw = [0], relax = True):
    losses = torch.pow(yhat-y, 2)
    if relax: losses = torch.clamp(losses, min=0.25) - 0.25
    if len(cw) > 1:
        sample_weights = torch.Tensor([cw.get(float(np.round(l))) for l in y])   
        return torch.sum(sample_weights.view(-1,1).cuda() * losses) / len(y)
    return torch.mean(losses)

# A custom assymmetric loss implemented in Numpy 
def custom_l2_loss(y, yhat, cw = [0], alpha = 0.25):
    alpha = (y * alpha) / 4
    loss = (y - yhat)**2 * (np.sign(y - yhat) + alpha)**2
    if len(cw) > 1:
        sw = [cw.get(float(np.round(l))) for l in y]
        return np.dot(loss, np.array(sw)) / len(y)
    return np.mean(loss)

# Smooth y, yhat with optimal smoothing params computed
def pred_smoother(x, y): 
    gauss_weights = gaussian(M = x[0].round(), std = x[1]) 
    gauss_weights = gauss_weights / gauss_weights.sum()
    yhat = filters.convolve1d(input = y[1], weights = gauss_weights)
    return mean_squared_error(y[0], y[1], sample_weight = y[2])

def smooth(y, yhat, sw):
    #opt = fmin(func = pred_smoother, x0 = np.asarray([15, 2]), args = ((y, yhat, sw), ))
    #gauss_weights = gaussian(M = opt[0].round(), std = opt[1].round())
    gauss_weights = gaussian(M = 20, std = 5)
    gauss_weights = gauss_weights / gauss_weights.sum()
    y = filters.convolve1d(input = y, weights = gauss_weights)
    yhat = filters.convolve1d(input = yhat, weights = gauss_weights)
    return y, yhat

def generate_eval(y, y_pred, sw = None):
    mae = mean_absolute_error(y, y_pred, sample_weight=sw)
    mse = mean_squared_error(y, y_pred, sample_weight=sw)
    acc = accuracy_score(np.round(y), np.round(y_pred), sample_weight=sw)
    res = [mae, mse, acc]
    print(res)
    return res