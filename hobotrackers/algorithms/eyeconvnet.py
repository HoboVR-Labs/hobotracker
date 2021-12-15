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


class BasicEncoder(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, desired_encoding_len, 13)

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


class DoubleLinear(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, hidden_dimension=256):
        super().__init__()
        self.lin1 = nn.Linear(in_dimensions, hidden_dimension)
        #self.bn1 = nn.BatchNorm1d(num_features=hidden_dimension)
        self.lin2 = nn.Linear(hidden_dimension, out_dimensions)

    def forward(self, x):
        x = self.lin1.forward(x)
        x = x/torch.max(x)
        x = self.lin2.forward(x)

        return x


class EyeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BasicEncoder()
        self.lin = DoubleLinear(96, int(16 * 2 ** 2))

    def forward(self, x):
        x, hs, ws = self.enc.forward(x)
        enc_with_pos = torch.flatten(torch.concat([x, hs, ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x


def eye_train_loop(loop_forever=True):
    # todo: add optional cv based distortions such as skew, rotation, or hue shifting
    for _, frame, eye_data in eye_iter(loop=True):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2:] / 255
        yield v, eye_data


def eye_into_to_floats(x):
    out_pint = pnums.PInt(0, 0, bits=16)
    new_shape = out_pint.tensor.shape
    out_pint.tensor = x.detach().cpu().numpy().reshape(new_shape)
    out_floats = [q / (2 ** 15) - 1 for q in out_pint.asfloat()],
    return out_floats


def eval_eye_iter(frame_iter, trained_model: EyeConvNet):
    for frame in frame_iter:
        v2v = combine_input_with_inversion(frame)
        v_new = cv_image_to_pytorch(v2v)
        x = trained_model(v_new)
        x = eye_into_to_floats(x)
        yield x


def train_eye_iter(sample_iter, model:EyeConvNet):
    loss = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for frame, eye_data in sample_iter:
        # adding inverted value onto input so we don't need biases to get useful stuff out of blank inputs:
        v2v = combine_input_with_inversion(frame)
        v_new = cv_image_to_pytorch(v2v)

        # get value and train
        optimizer.zero_grad()
        x = model(v_new)
        goal_enc = pnums.PInt(*[(q + 1) * 2 ** 15 for q in eye_data], bits=16)
        loss_val = loss(x, torch.FloatTensor(goal_enc.tensor.flatten()))
        loss_val.backward()
        optimizer.step()

        # yield predicted vs actual outputs. We can use this to determine when to stop training.
        yield {
            'Guess': eye_into_to_floats(x),
            'Actual': eye_data
        }


def simple_eval_iter(frame_iter, model_filename='eye_conv_net_1.torch'):
    model = EyeConvNet()
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    for x in eval_eye_iter(frame_iter, model):
        yield x