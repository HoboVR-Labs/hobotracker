# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

import torch
import time
from hobotrackers.algorithms.eyeconvnetpnum import EyeConvNetP, eye_train_loop, train_eye_iter_p

time_in_seconds = 3600
output_filename = 'eye_conv_net_1.torch'

t0 = time.time()

model = EyeConvNetP()

for d in train_eye_iter_p(eye_train_loop(), model):
    print(d)
    t1 = time.time()
    if t1 - t0 > time_in_seconds:
        break

torch.save(model.state_dict(), output_filename)
