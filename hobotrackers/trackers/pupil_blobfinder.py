# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>

"""Use blob tracking to find where a pupil, or large black object, is in an input video feed."""

from hobotrackers.algorithms.blobfinder import BlobFinder, ColorRange
from displayarray import display, read_updates
from displayarray.window import SubscriberWindows


def find_pupil_loop(cam, callback, color_range =ColorRange(2, 0, 0, 0, 61, 12, 112),  show=False):
    """
    Find where a pupil (or other large black circle) is in an input video feed.

    :param cam: int representing camera, camera hardware uid (if on linux), or filename
    :param callback: function to call once an image is done being scanned. Should take in center and radius floats.
    :param show: Whether or not we show the camera input and algorithm progress.
    :return:
    """
    if show:
        cam_read = display
        draw_find = True
    else:
        cam_read = read_updates
        draw_find = False

    b = BlobFinder(color_range=color_range, draw_find=draw_find)

    if isinstance(cam, SubscriberWindows):
        d = cam
    else:
        d = cam_read(cam, size=(9999, 9999), fps_limit=60)
    for up in d:
        if up:
            blob = b.find_blob(next(iter(up.values()))[0])
            if blob is not None:
                center, radius = blob
                callback(center, radius)
            else:
                callback(None, None)
