# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

from hobotrackers.trackers.pupil_blobfinder import find_pupil_loop
from resources import saccade_test


def print_callback(center, radius):
    if center is not None:
        print(f"center: {center}, radius: {radius}")
    else:
        print("blob not found. :(")


find_pupil_loop(saccade_test, print_callback, show=True)
