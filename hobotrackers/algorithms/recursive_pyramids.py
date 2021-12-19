import torch
import torch.nn.grad
import torchvision.transforms.functional
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Union, List, Sequence
import math as m
from torchvision.transforms import functional as FV
import warnings
import numpy as np
import sys
from torch.nn import functional as F


# from sparsepyramids.normal_conv import NormConv2d, NormConvTranspose2d
# from sparsepyramids.fixed_saturate_tensor import saturate_tensor, saturate_duplicates_tensor


class RecursivePyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            scale_val: float = m.sqrt(2),
            interpolation=FV.InterpolationMode.BILINEAR,
            min_size=3,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()
        if not isinstance(scale_val, float):
            raise TypeError(
                "scale_val should be a real number. Got {}".format(type(scale_val))
            )
        assert scale_val > 1.0, "scale_val needs to be greater than one."
        if not isinstance(min_size, (int, Sequence)) and not min_size is None:
            raise TypeError(
                "min_size should be int, sequence, or None. Got {}".format(
                    type(scale_val)
                )
            )
        if isinstance(min_size, (float, int)):
            min_size = [min_size, min_size]
        if isinstance(min_size, Sequence):
            if len(min_size) not in (1, 2):
                raise ValueError(
                    "If min_size is a sequence, it should have 1 or 2 values"
                )
            elif len(min_size) == 1:
                min_size = list(min_size) * 2
        self.min_size = min_size

        self.scale_val = scale_val
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def pyramidalize(self, input: Tensor):
        in_shape = list(input.shape[-2:])

        out = [input]

        while True:
            in_shape[0] = max(m.floor(in_shape[0] / self.scale_val), 1)
            in_shape[1] = max(m.floor(in_shape[1] / self.scale_val), 1)
            if self.min_size is None:
                if in_shape[0] <= 1 and in_shape[1] <= 1:
                    break
            else:
                if in_shape[0] <= self.min_size[0] or in_shape[1] <= self.min_size[1]:
                    break
            out.append(FV.resize(input, in_shape, self.interpolation))

        return out

    def pyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = []
            for i in input:
                out.append(self.pyramidalize_list(i))
            return out
        elif isinstance(input, Tensor):
            return self.pyramidalize(input)

    def forward(self, input: Tensor):
        out = self.pyramidalize_list(input)

        return out


def apply_func_to_nested_tensors(input, op, *args, min_size=3, **argv):
    if isinstance(min_size, (float, int)):
        min_size = [min_size, min_size]
    if isinstance(min_size, Sequence):
        if len(min_size) not in (1, 2):
            raise ValueError(
                "If min_size is a sequence, it should have 1 or 2 values"
            )
        elif len(min_size) == 1:
            min_size = list(min_size) * 2

    def apply_list(input: Union[Tensor, List[Tensor]]):
        nonlocal min_size
        if isinstance(input, list):
            out = []
            for i in input:
                if i.shape[-2] <= min_size[0] or i.shape[-1] <= min_size[1]:
                    break
                out.append(apply_list(i))
            return out
        elif isinstance(input, Tensor):
            return op(input, *args, **argv)

    return apply_list(input)


def full_max_pool_2d_nested(input, *args, **argv):
    def apply_list(input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = []
            for i in input:
                out.append(apply_list(i))
            return out
        elif isinstance(input, Tensor):
            return F.max_pool2d(input, input.shape[-2:], input.shape[-2:], *args, **argv)

    return apply_list(input)


def ravel_nested(input):
    partial_ravel = apply_func_to_nested_tensors(input, torch.ravel)

    def cat_nested(input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = None
            for i in input:
                if out is None:
                    out = cat_nested(i)
                else:
                    out = torch.cat([out, cat_nested(i)])
            return out
        elif isinstance(input, Tensor):
            return input

    return cat_nested(partial_ravel)
