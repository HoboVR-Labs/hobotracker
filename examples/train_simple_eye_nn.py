# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import torch
import time
from hobotrackers.algorithms.eyeconvnet import EyeConvNet, eye_train_loop, train_eye_iter

time_in_seconds = 3600
output_filename = 'eye_conv_net_1.torch'

t0 = time.time()

model = EyeConvNet()

for d in train_eye_iter(eye_train_loop(), model):
    print(d)
    t1 = time.time()
    if t1 - t0 > time_in_seconds:
        break

torch.save(model.state_dict(), output_filename)
