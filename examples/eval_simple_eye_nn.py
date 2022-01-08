# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

from displayarray import read_updates
from hobotrackers.algorithms.eyeconvnetpnum import eye_train_loop, simple_eval_iter_p


def grab_frames(cam):
    for d in read_updates(cam):
        if d:
            f = next(iter(d.values()))
            yield f


def eye_eval_loop():
    for frame, _ in eye_train_loop():
        yield frame


for x in simple_eval_iter_p(eye_eval_loop()):
    print(x)
