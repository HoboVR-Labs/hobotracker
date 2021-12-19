# SPDX-License-Identifier: GPL-2.0-only

# Copyright (C) 2020 Josh Miklos <josh.miklos@hobovrlabs.org>
import math

import torch
import time

from hobotrackers.algorithms.eyeconvnet_longer_conv import EyeConvNet5
from hobotrackers.algorithms.eyeconvnet_longer_lin import EyeConvNetTripLin
from hobotrackers.algorithms.eyeconvnet_no_bias import EyeConvNetNB
from hobotrackers.algorithms.eyeconvnet import EyeConvNet, eye_train_loop, train_eye_iter, eye_into_to_floats
from hobotrackers.algorithms.eyeconvnet_no_inv import EyeConvNetNoInv
from hobotrackers.algorithms.eyeconvnet_pyr import EyeConvNetPyr
from resources import eye_iter
from hobotrackers.util.cv_torch_helpers import cv_image_to_pytorch
from hobotrackers.util.general_nn_helpers import combine_input_with_inversion
import csv
from typing import Optional
import cv2


def train(model, export_csv=None, loops=1e6, verbose=True, lr=1e-4, inv=True):
    if export_csv is not None:
        csv_file = open(export_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['x_actual', 'y_actual', 'x_predicted', 'y_predicted',
                             'distance_x', 'distance_y', 'distance', 'total_dist', 'average_dist'])

    for i, d in enumerate(train_eye_iter(eye_train_loop(), model, lr=lr, inv=inv)):
        x = d['Guess']
        a = d['Actual']

        dist_x = abs(a[0] - x[0])
        dist_y = abs(a[1] - x[1])
        dist_xy = math.sqrt(dist_x ** 2 + dist_y ** 2)

        if export_csv is not None:
            csv_writer.writerow([*a, *x, dist_x, dist_y, dist_xy])

        if verbose:
            print(d)
        if i > loops:
            break

    if export_csv is not None:
        csv_file.close()

    return model


def eval(model, export_csv: Optional[str] = None, inv=True):
    if export_csv is not None:
        csv_file = open(export_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['x_actual', 'y_actual', 'x_predicted', 'y_predicted',
                             'distance_x', 'distance_y', 'distance', 'total_dist', 'average_dist'])
    max_i = 0
    total_dist = 0
    for i, frame, data in eye_iter(loop=False):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2:] / 255
        if inv:
            v2v = combine_input_with_inversion(v)
        else:
            v2v = v

        v_new = cv_image_to_pytorch(v2v)
        x = model(v_new)
        x = eye_into_to_floats(x)

        dist_x = abs(data[0] - x[0])
        dist_y = abs(data[1] - x[1])
        dist_xy = math.sqrt(dist_x ** 2 + dist_y ** 2)

        print(f'Prediction distance: {dist_xy}')

        max_i = i
        total_dist += dist_xy

        if export_csv is not None:
            csv_writer.writerow([*data, *x, dist_x, dist_y, dist_xy])

    avg_dist = total_dist / (max_i + 1)

    if export_csv is not None:
        csv_writer.writerow(['', '', '', '', '', '', '', total_dist, avg_dist])

    if export_csv is not None:
        csv_file.close()

    return avg_dist


best_avg = 0.4790339588270681


def test_eyeconvnet(capsys):
    t0 = time.time()
    with open('test_eyeconvnet.txt', 'w') as file:
        model = train(EyeConvNet(), 'EyeConvNet_train.csv', loops=1e5)
        t1 = time.time()
        file.write(f'train time: {t1-t0}')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}')

        file.write(f'avg dist: {avg_dist}')
        file.write(f'vs best avg: {best_avg - avg_dist}')


def test_eyeconvnet_no_inv(capsys):
    t0 = time.time()
    with open('test_eyeconvnet_no_inv.txt', 'w') as file:
        model = train(EyeConvNetNoInv(), 'EyeConvNet_no_inv_train.csv', loops=1e5, inv=False)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_no_inv_eval.csv', inv=False)
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')


def test_eyeconvnet_lr1e5(capsys):
    t0 = time.time()
    with open('test_eyeconvnet_lr1e5.txt', 'w') as file:
        model = train(EyeConvNet(), 'EyeConvNet_lr1e-5_train.csv', loops=1e5, lr=1e-5)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_lr1e-5_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')


def test_eyeconvnetnb(capsys):
    t0 = time.time()
    with open('test_eyeconvnetnb.txt', 'w') as file:
        model = train(EyeConvNetNB(), 'EyeConvNet_no_bias_train.csv', loops=1e5)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_no_bias_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')


def test_eyeconvnetpyr(capsys):
    t0 = time.time()
    with open('test_eyeconvnetpyr.txt', 'w') as file:
        model = train(EyeConvNetPyr(), 'EyeConvNet_pyr_train.csv', loops=1e5)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_pyr_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')


def test_eyeconvnet_longer_lin(capsys):
    t0 = time.time()
    with open('test_eyeconvnet_longer_lin.txt', 'w') as file:
        model = train(EyeConvNetTripLin(), 'EyeConvNet_longer_lin_train.csv', loops=1e5)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_longer_lin_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')


def test_eyeconvnet_longer_conv(capsys):
    t0 = time.time()
    with open('test_eyeconvnet_longer_conv.txt', 'w') as file:
        model = train(EyeConvNet5(), 'EyeConvNet_longer_conv_train.csv', loops=1e5)
        t1 = time.time()
        file.write(f'train time: {t1 - t0}\n')
        t0 = t1

        avg_dist = eval(model, 'EyeConvNet_longer_conv_eval.csv')
        t1 = time.time()
        file.write(f'eval time: {t1 - t0}\n')

        file.write(f'avg dist: {avg_dist}\n')
        file.write(f'vs best avg: {best_avg - avg_dist}\n')