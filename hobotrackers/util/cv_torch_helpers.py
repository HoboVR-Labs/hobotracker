# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import numpy as np
import torch


def cv_image_to_pytorch(image: np.ndarray, cuda: bool = True) -> torch.Tensor:
    torch_image = torch.from_numpy(image).float()
    if torch_image.ndim == 2:
        torch_image = torch_image[..., None]
    torch_image = torch_image.permute((2, 0, 1))
    torch_image = torch_image[None, ...]
    return torch_image