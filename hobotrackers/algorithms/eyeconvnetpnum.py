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


class TripleLinearP(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, hidden_dimension=256):
        super().__init__()
        self.lin1 = nn.Linear(in_dimensions, hidden_dimension)
        self.lin2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.lin3 = nn.Linear(hidden_dimension, out_dimensions)

    def forward(self, x):
        x = self.lin1.forward(x)
        x = x / torch.max(x)
        x = self.lin2.forward(x)
        x = x / torch.max(x)
        x = self.lin3.forward(x)

        return x


class PyrEncoderP(nn.Module):
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


class EyeConvNetP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PyrEncoderP()
        self.lin = TripleLinearP(480, int(16 * 2 ** 2))

    def forward(self, x):
        half_x = x[:, :, ::2, ::2]

        x, hs, ws = self.enc.forward(half_x)
        enc_with_pos = torch.flatten(torch.concat([*x, *hs, *ws], dim=0))
        x = self.lin.forward(enc_with_pos)

        return x


def eye_train_loop(loop_forever=True):
    # todo: add optional cv based distortions such as skew, rotation, or hue shifting
    for _, frame, eye_data in eye_iter(loop=True, shuffle=True):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2:] / 255
        yield v, eye_data


def eye_into_to_floats(x):
    out_pint = pnums.PInt(0, 0, bits=16)
    new_shape = out_pint.tensor.shape
    out_pint.tensor = x.detach().cpu().numpy().reshape(new_shape)
    out_floats = [q / (2 ** 15) - 1 for q in out_pint.asfloat()]
    return out_floats


def eval_eye_iter_p(frame_iter, trained_model: EyeConvNetP):
    for frame in frame_iter:
        v2v = combine_input_with_inversion(frame)
        v_new = cv_image_to_pytorch(v2v)
        x = trained_model(v_new)
        x = eye_into_to_floats(x)
        yield x


def pnum_loss(input:torch.Tensor, goal: pnums.PInt, base_loss = nn.SmoothL1Loss):
    """Gives a loss function so that large number difference have more loss than small ones."""
    # todo: move this to the pnums library

    loss = None

    input_size = torch.numel(input)
    goal_size = goal.tensor.size

    loss_fn = base_loss()

    assert input_size == goal_size, f"Input tensor size ({input_size}) should match goal size ({goal_size})."

    in_tensor = torch.reshape(input, goal.tensor.shape)

    for b in range(goal.bits):
        input_part = in_tensor[..., b]
        goal_part = goal.tensor[..., b]
        if loss is None:
            loss = loss_fn(input_part, torch.FloatTensor(goal_part)) / 2**b
        else:
            loss += loss_fn(input_part, torch.FloatTensor(goal_part)) / 2**b

    return loss


def train_eye_iter_p(sample_iter, model: EyeConvNetP, lr=1e-4, inv=True):
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
        goal_enc = pnums.PInt(*[(q + 1) * 2 ** 15 for q in eye_data], bits=16)
        loss_val = pnum_loss(x, goal_enc)
        loss_val.backward()
        optimizer.step()

        # yield predicted vs actual outputs. We can use this to determine when to stop training.
        yield {
            'Guess': eye_into_to_floats(x),
            'Actual': eye_data,
            'Loss': loss_val.detach().cpu().numpy()
        }


def simple_eval_iter_p(frame_iter, model_filename='eye_conv_net_1.torch'):
    model = EyeConvNetP()
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    for x in eval_eye_iter_p(frame_iter, model):
        yield x
