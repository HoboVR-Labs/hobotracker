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
crop_okawo_eye = crop.Crop((540,960,3), (540, 1440))

def eye_iter(loop = False):
    cam = cv2.VideoCapture(eye_avi)
    eye_data = np.load(eye_npy)
    while 1:
        ret, frame = cam.read()
        if not ret:
            print("vid dead")
            break

        current_frame = int(cam.get(cv2.CAP_PROP_POS_FRAMES)-1)
        last_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)-1)

        if current_frame>=last_frame:
            if loop:
                cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                break

        yield current_frame, frame, eye_data[current_frame, ...]

    cam.release()
