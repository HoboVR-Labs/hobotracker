from resources import okawo_3, crop_okawo_eye
from displayarray import display
from displayarray.effects import crop
from hobotrackers.algorithms.blobfinder import ColorRange
from hobotrackers.trackers.pupil_blobfinder import find_pupil_loop

crop_eye_further = crop.Crop((360, 620, 3), (180, 310))

cam = display(okawo_3, callbacks=[crop_okawo_eye, crop_eye_further])


def print_callback(center, radius):
    if center is not None:
        print(f"center: {center}, radius: {radius}")
    else:
        print("blob not found. :(")


find_pupil_loop(cam, print_callback, ColorRange(2, 105, 31, 167, 61, 47, 28), show=True)
