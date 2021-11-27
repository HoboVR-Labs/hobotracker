"""
pyvr calibrate.

Usage:
    hobotrackers calibrate [options]

Options:
   -h, --help
   -c, --camera <camera>        Source of the camera to use for calibration [default: 0]
   -r, --resolution <res>       Input resolution in width and height [default: -1x-1]
   -n, --n_masks <n_masks>      Number of masks to calibrate [default: 1]
   -l, --load_from_file <file>  Load previous calibration settings [default: ranges.pickle]
   -s, --save <file>            Save calibration settings to a file [default: ranges.pickle]
   -L, --linux                  Linux based v4l2 settings.
"""

from docopt import docopt
import sys
from hobotrackers.util.manual_color_mask_calibration import manual_calibration
from hobotrackers import __version__


def main(argv=None):
    """Calibrate entry point."""
    # allow calling from both python -m and from pyvr:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2 or (len(sys.argv) > 1 and sys.argv[1] != "calibrate"):
        argv = ["calibrate"] + argv

    args = docopt(__doc__, version=f"hobotrackers version {__version__}", argv=argv)

    width, height = args["--resolution"].split("x")

    if args["--camera"].isdigit():
        cam = int(args["--camera"])
    elif args["--camera"] == "saccade_test":
        from resources import saccade_test

        cam = saccade_test
    else:
        cam = args["--camera"]

    manual_calibration(
        cam=cam,
        num_colors_to_track=int(args["--n_masks"]),
        frame_width=int(width),
        frame_height=int(height),
        load_file=args["--load_from_file"],
        save_file=args["--save"],
        linux=args["--linux"],
    )


if __name__ == "__main__":
    argv = ["calibrate", "-c", "saccade_test"]
    main(argv)
