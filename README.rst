hobotrackers
============

A library containing various tracking algorithms and implementations.

Current Implementations
----------------------------------

blobfinding for eye tracking:

::

    from hobotrackers.trackers.pupil_blobfinder import find_pupil_loop
    from hobotrackers.resources import saccade_test


    def print_callback(center, radius):
        if center is not None:
            print(f"center: {center}, radius: {radius}")
        else:
            print("blob not found. :(")


    find_pupil_loop(saccade_test, print_callback, show=True)


Installation
------------

hobotrackers will be distributed on `PyPI <https://pypi.org>`__ as a
universal wheel in Python 3.6+ and PyPy.

::

    $ pip install hobotrackers

Usage
-----

API has not been generated yet.

See tests and examples for example usage.

License
-------

hobotrackers is distributed under the terms of

-  `GPL 2.0 License <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>`__

