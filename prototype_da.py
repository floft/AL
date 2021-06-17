#!/usr/bin/python

# Contains functions to perform augmentation of wearable sensor data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.

"""
Call this function from within extract_features in al.py. Each time the end of a
window is reached and the location is valid, features.create_point is called to
extract a feature vector. The feature vector is added to xdata and the
corresponding label is added to ydata.

When cf.data_augmentation == True and cf.mode == config.MODE_TRAIN_MODEL, the
extract_features function will now also call augment_points. Function
augment_points will create a set of new data windows using the two strategies
listed below, and return the set of new data windows. The extract_features
function will call features.create_point on each new data window. The feature
vector for each new window will be added to xdata. The same label that was
assigned to the original, true data window will be added to ydata.

Augmentation strategies
1. Jitter to location data (within closeness threshold), same label, same times
2. Interpolate between time points to create new data, same label, same times
"""

import random

def augment_points(st, dt, person_stats):
  new_points = []

  # create new points using jitter strategy
  num_jitter = 100
  for i in range(num_jitter): 
    new_point = duplicate(st.date, st.time, st.yaw, ...)
    new_point.latitude = add_location_jitter(st.latitude)
    new_point.longitude = add_location_jitter(st.latitude)
    new_point.altitude = add_location_jitter(st.latitude)
    new_points.append(new_point)

  # create new points using interpolation strategy
  num_interpolate = 100
  n = cf.samplesize
  new_points_interpolate = np.zeros((n, cf.numsensors))
  inc = 1.0 / num_interpolate
  for i in range(num_interpolate):
    for j in range(n-1):  # sequence through all points in the window
        new_points_interpolate[j].date = st[j].date
        new_points_interpolate[j].time = st[j].time
        for k in range(cf.numsensors):   # k = yaw, pitch, roll, ...
          # for all sensor values, create a number point that is
          # somewhere between the value at time i and the value at time i+1
          new_points_interpolate[j].k = \
            interpolate_value(st[j].k, st[j+1].k, inc, num_interpolate)
        # at the end of the window just copy the last point
        new_points_interpolate[n-1].date = st[n-1].date
        new_points_interpolate[n-1].time = st[n-1].time
        for k in range(cf.numsensors):   # k = yaw, pitch, roll, ...
          new_points_interpolate[n-1].k = st[n-1].k
        new_points.append(new_points_interpolate)
  return new_points


def add_location_jitter(value):
  value = random.uniform(value - 0.00025, value + 0.00025)
  return value


def interpolate_value(value1, value2, inc, num_interpolate):
  value = ((value * inc) + (value1 * (1.0 - inc))) / num_interpolate
  return value
