#!/usr/bin/python
# Augment training data by making copies of an initial window with jitter and
# interpolation values.
#
# Written by Bryan Minor, Washington State University.
#
# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.
from typing import List

import numpy as np

from al import AL, DataWindow
from config import Config


def augment_points(orig_window: AL, conf: Config) -> List[DataWindow]:
    """
    Create augmented data from the sensor data window stored on AL.
    This consists of two kinds of augmentation:
     - Copies of the data with jitter added to the location sensor values
     - Interpolation "slices" at specific offsets for all sensors throughout
       the window

    TODO: Switch to take in a DataWindow object when AL inherits from that (or,
     better yet, remove the sensor tracking from AL to DataWindow completely)

    Parameters
    ----------
    orig_window : AL
        The original sensor data window (stored on an AL object for now)
    conf : Config
        The config we are currently using

    Returns
    -------
    List[DataWindow]
        The augmented window copies that can be used to create new feature
        vectors
    """

    new_windows = add_location_jitter(orig_window, conf)

    new_windows.extend(create_interpolation_windows(orig_window, conf))

    return new_windows


def add_location_jitter(orig_window: AL, conf: Config) -> List[DataWindow]:
    """
    Add location jitter by adding uniform noise (within the confines of
    +/- `conf.da_jitter_limit`) to the `lat`, `lon`, and `alt` sensor values.
    Make `conf.da_num_jitter` copies this way and return them.
    """

    new_windows = list()  # type: List[DataWindow]

    for _ in range(conf.da_num_jitter):
        # Start with a copy of the original window:
        new_window = DataWindow(conf, orig_window.location)
        new_window.copy_from(orig_window)  # copy data from the original window

        # Now add the jitter:
        new_window.latitude = add_jitter_to_list(
            new_window.latitude,
            conf.da_jitter_limit
        )

        new_window.longitude = add_jitter_to_list(
            new_window.longitude,
            conf.da_jitter_limit
        )

        new_window.altitude = add_jitter_to_list(
            new_window.altitude,
            conf.da_jitter_limit
        )

        # Add the window to the list:
        new_windows.append(new_window)

    return new_windows


def add_jitter_to_list(data: List[float], jitter_limit: float) \
        -> List[float]:
    """Add uniform jitter within the specified limits to the sensor data."""

    jitter = np.random.uniform(-jitter_limit, jitter_limit, len(data))

    return list(data + jitter)


def create_interpolation_windows(orig_win: AL, conf: Config) \
        -> List[DataWindow]:
    """
    Create `conf.da_num_interpolations - 2` copies of the data, each with the
    values interpolated at a certain "slice" between each successive pair of
    sensor data values. That is, each will have the values interpolated
    `i / da_num_interpolations` of the way between points across all points and
    sensors.

    In this way, we generate windows with the same number of sample points, but
    where each has points that are a slightly shifted mix of interpolation
    between the values in the original data.

    Note that we skip the cases where `i = 0` and `i = da_num_interpolations` to
    avoid creating essentially duplicates of the original window. This results
    in there actually being only `da_num_interpolations - 2` new windows.
    """

    new_windows = list()  # type: List[DataWindow]

    # Iterate through 0 < i < da_num_interpolations:
    for i in range(1, conf.da_num_interpolations):
        # Compute the fraction of the way through to slice interpolation:
        interp_frac = i / conf.da_num_interpolations

        # Now create a new window by taking the same interpolation fraction
        # at all sensor data points:
        new_win = DataWindow(conf, orig_win.location)

        new_win.yaw = get_interp_slice(orig_win.yaw, interp_frac)
        new_win.pitch = get_interp_slice(orig_win.pitch, interp_frac)
        new_win.roll = get_interp_slice(orig_win.roll, interp_frac)

        new_win.rotx = get_interp_slice(orig_win.rotx, interp_frac)
        new_win.roty = get_interp_slice(orig_win.roty, interp_frac)
        new_win.rotz = get_interp_slice(orig_win.rotz, interp_frac)

        new_win.accx = get_interp_slice(orig_win.accx, interp_frac)
        new_win.accy = get_interp_slice(orig_win.accy, interp_frac)
        new_win.accz = get_interp_slice(orig_win.accz, interp_frac)
        new_win.acctotal = get_interp_slice(orig_win.acctotal, interp_frac)

        new_win.latitude = get_interp_slice(orig_win.latitude, interp_frac)
        new_win.longitude = get_interp_slice(orig_win.longitude, interp_frac)
        new_win.altitude = get_interp_slice(orig_win.altitude, interp_frac)

        new_win.course = get_interp_slice(orig_win.course, interp_frac)
        new_win.speed = get_interp_slice(orig_win.speed, interp_frac)

        new_win.hacc = get_interp_slice(orig_win.hacc, interp_frac)
        new_win.vacc = get_interp_slice(orig_win.vacc, interp_frac)

        # Compute the new min/max lat/lon:
        new_win.minlat = min(new_win.latitude)
        new_win.maxlat = max(new_win.latitude)

        new_win.minlong = min(new_win.longitude)
        new_win.maxlong = max(new_win.longitude)

        new_windows.append(new_win)

    return new_windows


def get_interp_slice(
        data: List[float],
        interp_frac: float
) -> List[float]:
    """
    Make an interpolation slice for the given sensor data. This will all the
    interpolated values at `interp_frac` of the way between subsequent original
    sensor values. (That is, use `1 - interp_frac` of the preceding point, and
    `interp_frac` of the succeeding point to create each new point.)

    For the last point, just repeat the last original point.

    Parameters
    ----------
    data : List[float]
        Original values for a particular sensor to interpolate
    interp_frac : float
        Fraction of the space between points to interpolate

    Returns
    -------
    List[float]
        The new interpolation slice of the sensor values.
    """

    new_data = list()  # type: List[float]

    # Loop through the spaces between each pair of points in the data:
    for j in range(len(data) - 1):
        # Interpolate interp_frac of the way between these two points
        # That is, use an average of the two, weighted by the interp_frac
        new_data.append((1 - interp_frac) * data[j] + interp_frac * data[j+1])

    # Use the original last value as the new last value:
    new_data.append(data[-1])

    return new_data
