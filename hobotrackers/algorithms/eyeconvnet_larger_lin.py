import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.algorithms.eyeconvnet import PyrEncoder
from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter
from hobotrackers.algorithms.recursive_pyramids import RecursivePyramidalize2D, apply_func_to_nested_tensors, \
    full_max_pool_2d_nested
import cv2


class BigTripleLinear(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, hidden_dimension=512):
        super().__init__()
        self.lin1 = nn.Linear(in_dimensions, hidden_dimension)
        self.lin2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.lin3 = nn.Linear(hidden_dimension, out_dimensions)

    def forward(self, x):
        x = self.lin1.forward(x)
        x = x/torch.max(x)
        x = self.lin2.forward(x)
        x = x / torch.max(x)
        x = self.lin3.forward(x)

        return x


class EyeConvNetLargerLin(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PyrEncoder()
        self.lin = BigTripleLinear(576, int(16 * 2 ** 2))

    def forward(self, x):
        half_x = x[:, :, ::2, ::2]

        x, hs, ws = self.enc.forward(half_x)
        enc_with_pos = torch.flatten(torch.concat([*x, *hs, *ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x
