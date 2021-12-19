# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.algorithms.eyeconvnet import DoubleLinear
from hobotrackers.algorithms.recursive_pyramids import RecursivePyramidalize2D, apply_func_to_nested_tensors, \
    full_max_pool_2d_nested
from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter


class PyrEncoder(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.pyr = RecursivePyramidalize2D()
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, desired_encoding_len, 3, padding=1)

    def forward(self, x):
        x = self.pyr.forward(x)

        x = apply_func_to_nested_tensors(x, self.conv1.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv2.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv3.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv4.forward)

        x = full_max_pool_2d_nested(x, 0, 1, False, True)

        x, x_ind = zip(*x)

        h = [xs.shape[-2] for xs in x]
        w = [xs.shape[-1] for xs in x]

        ih = [torch.div(x_inds, ws, rounding_mode='floor') for x_inds, ws in zip(x_ind, w)]
        iw = [x_inds - ihs for x_inds, ihs in zip(x_ind, ih)]
        hs = [ihs/hs for ihs, hs in zip(ih, h)]
        ws = [iws/ws for iws, ws in zip(iw, w)]

        # these are indices and cannot be backpropogated to
        # hs.detach_()
        # ws.detach_()

        return x, hs, ws


class EyeConvNetPyr(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PyrEncoder()
        self.lin = DoubleLinear(768, int(16 * 2 ** 2))

    def forward(self, x):
        x, hs, ws = self.enc.forward(x)
        enc_with_pos = torch.flatten(torch.concat([*x, *hs, *ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x
