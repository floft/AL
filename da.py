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
