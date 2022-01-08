# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

"""Videos and other resources that can be used with examples or tests."""

import os
from displayarray.effects import crop
import cv2
import numpy as np

# install these files from the drive

dir_path = os.path.dirname(os.path.realpath(__file__))

saccade_test = dir_path + os.sep + "saccade_test.webm"

okawo_1 = dir_path + os.sep + "2021-12-12_18-43-17.mp4"
okawo_2 = dir_path + os.sep + "2021-12-12_18-58-57.mp4"
okawo_3 = dir_path + os.sep + "2021-12-12_19-13-17.mp4"

eye_avi = dir_path + os.sep + "left_eye_only_rgb.avi"
eye_npy = dir_path + os.sep + "left_eye_only_rgb_gt.npy"
crop_okawo_eye = crop.Crop((540, 960, 3), (540, 1440))


def eye_iter(loop=False, shuffle=False, random=False):
    """
    loop: use to train forever on a small vid
    shuffle: use to improve training
    random: use instead of shuffle for extremely large videos
    """
    cam = cv2.VideoCapture(eye_avi)
    eye_data = np.load(eye_npy)

    assert (not random) or (not shuffle), "eye_iter can either shuffle frames or select them randomly, not both."

    h = 0
    next_frame = 0

    if shuffle:
        slices = np.arange(eye_data.shape[0], dtype=int)

        np.random.shuffle(slices)
        print("data shuffled:")
        print(slices)

        while 1:
            ret, frame = cam.read()
            if not ret:
                print("vid dead")
                break

            cam.set(cv2.CAP_PROP_POS_FRAMES, slices[h])
            yield h, frame, eye_data[h, ...]

            h += 1
            if h >= eye_data.shape[0]:
                if loop:
                    h = 0
                    if shuffle:
                        np.random.shuffle(slices)
                        print("data shuffled:")
                        print(slices)
                else:
                    break
    else:
        while 1:
            ret, frame = cam.read()
            if not ret:
                print("vid dead")
                break

            frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

            yield next_frame, frame, eye_data[next_frame, ...]

            if random:
                next_frame = np.random.randint(0, frame_count)
                cam.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

            h += 1
            if h >= frame_count:
                if loop:
                    if not random:
                        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    break

    cam.release()
