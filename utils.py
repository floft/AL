#!/usr/bin/python

# Contains utility functions.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


from datetime import datetime

from scipy import signal

import config

cf = config.Config(description='')


def get_datetime(date, dt_time):
    """ Input is two strings representing a date and a time with the format
    YYYY-MM-DD HH:MM:SS.ms. This function converts the two strings to a single
    datetime.datetime() object.
    """
    dt_str = date + ' ' + dt_time
    if '.' in dt_str:  # Remove optional millsecond precision
        dt_str = dt_str.split('.', 1)[0]
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return dt


def clean_range(value, low, high):
    """ Clean up values that fall outside of a specified range.
    Replace outliers with the range min or max.
    """
    if value > high:
        return high
    elif value < low:
        return low
    else:
        return value


def clean(value, low, high):
    """ Clean up values that fall outside of definable range. Replace outliers
    with mean for the feature.
    """
    if low <= value <= high:
        return value
    else:
        return (high + low) / 2.0


def butter_lowpass_filter(data):
    """ Apply a Butterworth low-pass filter with cutoff 0.3Hz to accelerometer
    and gyroscope data. Courtesy Guillaume Chevalier.
    """
    cutoff_frequency = 0.3
    nyq_freq = float(cf.samplerate) / 2.0
    normal_cutoff = float(cutoff_frequency) / nyq_freq
    b, a = signal.butter(4, normal_cutoff, btype='lowpass')
    y = signal.filtfilt(b, a, data, padlen=0)
    return y


def process_entry(line):
    """ Parse a single input line containing a sensor reading.
    The format is "date time sensorname sensorname value <activitylabel|0>".
    """
    x = str(str(line).strip()).split(' ', 5)
    x[5] = x[5].replace(' ', '_')
    return x
