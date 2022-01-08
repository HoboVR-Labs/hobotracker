import cv2
import pnums
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
from resources import eye_iter
from hobotrackers.algorithms.recursive_pyramids import RecursivePyramidalize2D, apply_func_to_nested_tensors, \
    full_max_pool_2d_nested
import cv2


class TripleLinear(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, hidden_dimension=256):
        super().__init__()
        self.lin1 = nn.Linear(in_dimensions, hidden_dimension)
        self.lin2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.lin3 = nn.Linear(hidden_dimension, hidden_dimension)
        self.lin4 = nn.Linear(hidden_dimension, out_dimensions)  # added for converting to float function

    def forward(self, x):
        x = self.lin1.forward(x)
        x = x / torch.max(x)
        x = self.lin2.forward(x)
        x = x / torch.max(x)
        x = self.lin3.forward(x)
        x = x / torch.max(x)
        x = self.lin4.forward(x)

        return x


class PyrEncoder(nn.Module):
    DEFAULT_ENCODING_LEN = 32

    def __init__(self, desired_encoding_len=DEFAULT_ENCODING_LEN):
        super().__init__()
        self.pyr = RecursivePyramidalize2D()
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, desired_encoding_len, 13, padding=1)

    def forward(self, x):
        x = self.pyr.forward(x)

        x = apply_func_to_nested_tensors(x, self.conv1.forward, min_size=13)
        x = apply_func_to_nested_tensors(x, self.pool.forward, min_size=13)
        x = apply_func_to_nested_tensors(x, self.conv2.forward, min_size=13)
        x = apply_func_to_nested_tensors(x, self.pool.forward, min_size=13)
        x = apply_func_to_nested_tensors(x, self.conv3.forward, min_size=13)
        x = apply_func_to_nested_tensors(x, self.conv4.forward)

        x = full_max_pool_2d_nested(x, 0, 1, False, True)

        x, x_ind = zip(*x)

        h = [xs.shape[-2] for xs in x]
        w = [xs.shape[-1] for xs in x]

        ih = [torch.div(x_inds, ws, rounding_mode='floor') for x_inds, ws in zip(x_ind, w)]
        iw = [x_inds - ihs for x_inds, ihs in zip(x_ind, ih)]
        hs = [ihs / hs for ihs, hs in zip(ih, h)]
        ws = [iws / ws for iws, ws in zip(iw, w)]

        return x, hs, ws


class EyeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PyrEncoder()
        self.lin = TripleLinear(480, 2)

    def forward(self, x):
        half_x = x[:, :, ::2, ::2]

        x, hs, ws = self.enc.forward(half_x)
        enc_with_pos = torch.flatten(torch.concat([*x, *hs, *ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x


def eval_eye_iter_f(frame_iter, trained_model: EyeConvNet):
    for frame in frame_iter:
        v2v = combine_input_with_inversion(frame)
        v_new = cv_image_to_pytorch(v2v)
        x = trained_model(v_new)
        x = x.detach().cpu().numpy()
        yield x


def train_eye_iter_f(sample_iter, model: EyeConvNet, lr=1e-4, inv=True):
    loss = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr)

    for frame, eye_data in sample_iter:
        # adding inverted value onto input so we don't need biases to get useful stuff out of blank inputs:
        if inv:
            v2v = combine_input_with_inversion(frame)
        else:
            v2v = frame
        v_new = cv_image_to_pytorch(v2v)

        # get value and train
        optimizer.zero_grad()
        x = model(v_new)
        loss_val = loss(x, torch.FloatTensor(eye_data))
        loss_val.backward()
        optimizer.step()

        # yield predicted vs actual outputs. We can use this to determine when to stop training.
        yield {
            'Guess': x,
            'Actual': eye_data,
            'Loss': loss_val.detach().cpu().numpy()
        }


def simple_eval_iter_f(frame_iter, model_filename='eye_conv_net_1.torch'):
    model = EyeConvNet()
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    for x in eval_eye_iter_f(frame_iter, model):
        yield x
