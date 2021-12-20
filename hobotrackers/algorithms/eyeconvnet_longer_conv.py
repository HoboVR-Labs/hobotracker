import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.algorithms.eyeconvnet import TripleLinear
from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter
from hobotrackers.algorithms.recursive_pyramids import RecursivePyramidalize2D, apply_func_to_nested_tensors, \
    full_max_pool_2d_nested
import cv2


class PyrEncoderLongerConv(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.pyr = RecursivePyramidalize2D()
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, desired_encoding_len, 3, padding=1)

    def forward(self, x):
        x = self.pyr.forward(x)

        x = apply_func_to_nested_tensors(x, self.conv1.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv2.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv3.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv4.forward)
        x = apply_func_to_nested_tensors(x, self.pool.forward)
        x = apply_func_to_nested_tensors(x, self.conv5.forward)

        x = full_max_pool_2d_nested(x, 0, 1, False, True)

        x, x_ind = zip(*x)

        h = [xs.shape[-2] for xs in x]
        w = [xs.shape[-1] for xs in x]

        ih = [torch.div(x_inds, ws, rounding_mode='floor') for x_inds, ws in zip(x_ind, w)]
        iw = [x_inds - ihs for x_inds, ihs in zip(x_ind, ih)]
        hs = [ihs/hs for ihs, hs in zip(ih, h)]
        ws = [iws/ws for iws, ws in zip(iw, w)]

        return x, hs, ws


class EyeConvNetLongerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PyrEncoderLongerConv()
        self.lin = TripleLinear(384, int(16 * 2 ** 2))

    def forward(self, x):
        half_x = x[:, :, ::2, ::2]

        x, hs, ws = self.enc.forward(half_x)
        enc_with_pos = torch.flatten(torch.concat([*x, *hs, *ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x
