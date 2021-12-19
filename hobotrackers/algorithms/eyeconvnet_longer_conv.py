# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.algorithms.eyeconvnet import DoubleLinear
from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter



class BasicEncoder5(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.conv5 = nn.Conv2d(32, desired_encoding_len, 3)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool.forward(x)
        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = self.conv3.forward(x)
        x = self.pool.forward(x)
        x = self.conv4.forward(x)
        x = self.pool.forward(x)
        x = self.conv5.forward(x)
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


class EyeConvNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BasicEncoder5()
        self.lin = DoubleLinear(96, int(16 * 2 ** 2))

    def forward(self, x):
        x, hs, ws = self.enc.forward(x)
        enc_with_pos = torch.flatten(torch.concat([x, hs, ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x