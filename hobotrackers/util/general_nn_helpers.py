# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import numpy as np


def combine_input_with_inversion(input_frame):
    framei = (1.0 - input_frame)
    v2v = np.concatenate((input_frame, framei), axis=2)
    return v2v