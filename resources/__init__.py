# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

"""Videos and other resources that can be used with examples or tests."""

import os
from displayarray.effects import crop

dir_path = os.path.dirname(os.path.realpath(__file__))

saccade_test = dir_path + os.sep + "saccade_test.webm"

okawo_1 = dir_path + os.sep + "2021-12-12_18-43-17.mp4"
okawo_2 = dir_path + os.sep + "2021-12-12_18-58-57.mp4"
okawo_3 = dir_path + os.sep + "2021-12-12_19-13-17.mp4"

crop_okawo_eye = crop.Crop((540,960,3), (540, 1440))
