# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter


class BasicEncoderNB(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 3, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, bias=False)
        self.conv3 = nn.Conv2d(8, 16, 3, bias=False)
        self.conv4 = nn.Conv2d(16, desired_encoding_len, 13, bias=False)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool.forward(x)
        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = self.conv3.forward(x)
        x = self.pool.forward(x)
        x = self.conv4.forward(x)
        h, w = x.shape[-2:]
        x, x_ind = F.max_pool2d(x, x.shape[-2:], x.shape[-2:], 0, 1, False, True)

        ih = torch.div(x_ind,w, rounding_mode='floor')
        iw = x_ind - ih
        hs = ih/h
        ws = iw/w

        # these are indices and cannot be backpropogated to
        # hs.detach_()
        # ws.detach_()

        return x, hs, ws


class DoubleLinearNB(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, hidden_dimension=256):
        super().__init__()
        self.lin1 = nn.Linear(in_dimensions, hidden_dimension, bias=False)
        #self.bn1 = nn.BatchNorm1d(num_features=hidden_dimension)
        self.lin2 = nn.Linear(hidden_dimension, out_dimensions, bias=False)

    def forward(self, x):
        x = self.lin1.forward(x)
        x = x/torch.max(x)
        x = self.lin2.forward(x)

        return x


class EyeConvNetNB(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BasicEncoderNB()
        self.lin = DoubleLinearNB(96, int(16 * 2 ** 2))

    def forward(self, x):
        x, hs, ws = self.enc.forward(x)
        enc_with_pos = torch.flatten(torch.concat([x, hs, ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x