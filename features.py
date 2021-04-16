#!/usr/bin/python

# Contains functions to calculate features from sensor data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os
import datetime
import math
from typing import Tuple

import joblib
import numpy as np
from scipy.stats import moment

import config
import person
import utils


def mean_absolute_deviation(data, axis=None):
    """ Compute mean absolute deviation of data.
    Input data is an array with an optional specified axis of computation.
    """
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def median_absolute_deviation(data, axis=None):
    """ Compute median absolute deviation of data.
    Input data is an array with an optional specified axis of computation.
    """
    return np.median(np.absolute(data - np.median(data, axis)), axis)


def zero_crossings(data, median):
    """ Compute zero crossings of the input array of data. Zero crossings is
    computed as the number of times the data value crosses the median as the
    sequence is traversed from beginning to end.
    """
    rel = 0
    count = 0
    for x in data:
        if x < median:
            if rel > 0:
                count += 1
            rel = -1
        elif x > median:
            if rel < 0:
                count += 1
            rel = 1
    return count


def mean_crossings(data, mean):
    """ Compute zero crossings of the input array of data. Mean crossings is
    computed as the number of times the data value crosses the mean as the
    sequence is traversed from beginning to end.
    """
    rel = 0
    count = 0
    for x in data:
        if x < mean:
            if rel > 0:
                count += 1
            rel = -1
        elif x > mean:
            if rel < 0:
                count += 1
            rel = 1
    return count


def moments(data):
    """ Calculate the 1st through 4th moments about the mean for the
    input array of data. These describe the shape of the data.
    """
    m1 = moment(data, 1)
    m2 = moment(data, 2)
    m3 = moment(data, 3)
    m4 = moment(data, 4)
    return m1, m2, m3, m4


def fft_features(data):
    """ Calculate the periodic components that express the input array of data
    using a Discrete Fourier Transform.
    """
    fft_feature = np.fft.fft(data)
    ceps = fft_feature[0].real
    lceps = len(fft_feature)
    psd = []
    psdtotal = 0.0
    energy = 0.0
    for i in range(lceps):
        r = fft_feature[i].real
        energy += r
        value = (r * r) / float(lceps) + 1e-08
        psd.append(value)
        psdtotal += value
    entropy = 0.0
    for i in range(lceps):
        entropy -= psd[i] * math.log(psd[i])
    msignal = np.mean(fft_feature).real
    vsignal = np.var(fft_feature).real
    return ceps, entropy, energy, msignal, vsignal


def interquartile_range(data):
    """ Calculate the interquartile range of the input array of data.
    This is the difference in values between the 25% value and 75% value of
    the sorted array of values.
    """
    newlist = sorted(data)
    num = len(newlist)
    x = newlist[(int(num / 4))]
    y = newlist[(int((3 * num) / 4))]
    return y - x


def skewness(data, mean):
    """ Calculate the amount of skewness, or leaning toward low or high values,
    in the input array of data.
    """
    sum1 = 0
    sum2 = 0
    for d in data:
        sum1 += (d - mean) ** 3
        sum2 += (d - mean) ** 2
    n1 = sum1 / len(data)
    n2 = sum2 / len(data)
    n2 = n2 ** 1.5
    if n2 == 0:
        return 0.0
    else:
        return n1 / n2


def kurtosis(data, mean, std):
    """ Calculate the amount of kurtosis, or peakedness/flatness,
    in the input array of data.
    """
    sum1 = 0
    for d in data:
        sum1 += (d - mean) ** 4
    n1 = sum1 / len(data)
    n2 = std ** 4
    if n2 == 0:
        return -3.0
    else:
        return (n1 / n2) - 3.0


def signal_energy(data):
    """ Calculate the signal energy of the data array. Because the data are
    discrete-time, energy is defined as the sum of the squared values.
    """
    signal_sum = 0
    for d in data:
        signal_sum += d ** 2
    return signal_sum


def log_signal_energy(data):
    """ Calculate the log signal energy of the data array. This is defined as
    the sum of log base 10 of the squared values.
    """
    signal_sum = 0
    for d in data:
        d2 = d * d
        if d2 != 0:
            signal_sum += math.log(d2, 10)
    return signal_sum


def signal_magnitude_area(x, y, z):
    """ Calculate the signal magnitude area of the input data array.
    """
    signal_sum = 0
    for i in range(len(x)):
        signal_sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return signal_sum


def correlation(x, y, mx, my):
    """ Two dimensions of data are input together with their means. This function
    calculates the correlation between the two data dimensions.
    """
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(len(x)):
        sum1 += (x[i] - mx) * (y[i] - my)
        sum2 += (x[i] - mx) ** 2
        sum3 += (y[i] - my) ** 2
    sum4 = sum2 * sum3
    if sum4 == 0:
        return 0.0
    sum4 = math.sqrt(sum4)
    return sum1 / sum4


def autocorrelation(x, mean):
    """ Calculate autocorrelation, or correlation of a data sequence with
    itself at different points in the sequence.
    """
    sum1 = 0
    sum2 = 0
    for i in range(1, len(x)):
        sum1 += (x[i - 1] - mean) * (x[i] - mean)
        sum2 += (x[i] - mean) ** 2
    # Check for no change case.
    if sum2 == 0 or len(x) <= 1:
        return 0.0
    sum1 /= (len(x) - 1)
    sum2 /= len(x)
    return sum1 / sum2


def heading_change_rate(course, distance):
    """ The input is sequences of course values and distance values
    (usually provided by location services). This function calculates the number
    of points in the sequence that indicate a change in direction, divided by
    the distance covered during the sequence.
    """
    if distance == 0:  # Avoid divide by zero
        return 0.0
    total = 0
    prevc = course[0]
    for c in course:
        if (prevc != c) and (c != -1.0):
            total += 1
            prevc = c
    return float(total) / distance


def stop_rate(latitude, longitude, distance):
    """ The input is sequences of latitude, longitude, and distance values
    (usually provided by location services). This function calculates the number
    of points in the sequence that have no change in location, divided by
    the distance covered during the sequence.
    """
    if distance == 0:  # Avoid divide by zero
        return 0.0
    total = 0
    prevlat = latitude[0]
    prevlong = longitude[0]
    for i in range(len(latitude)):
        if (prevlat == latitude[i]) and (prevlong == longitude[i]):
            total += 1
        prevlat = latitude[i]
        prevlong = longitude[i]
    return float(total) / distance


def trajectory(latitude, longitude):
    """ Calculate the overall trajectory indicated by the input sequences of
    latitude and longitude values.
    """
    minl = min(latitude)
    maxl = max(latitude)
    difflat = maxl - minl
    if difflat == 0.0:  # Avoid divide by zero
        return -1.57079633
    minl = min(longitude)
    maxl = max(longitude)
    difflong = maxl - minl
    slopepercent = difflong / difflat
    return math.atan(slopepercent)


def time_between_peaks(x):
    """ Compute the time (number of values) that elapses between the two
    peak values in the sequence.
    """
    n = len(x)
    y = np.argsort(x)
    return abs(y[n - 1] - y[n - 2])


def twod_features(al):
    """ Create a list of statistical features that process two dimensions
    at a time.
    """
    flist = list()
    flist.append(correlation(al.yaw,
                             al.pitch,
                             np.mean(al.yaw),
                             np.mean(al.pitch)))
    flist.append(np.mean(al.yaw) - np.mean(al.pitch))
    flist.append(correlation(al.yaw,
                             al.roll,
                             np.mean(al.yaw),
                             np.mean(al.roll)))
    flist.append(np.mean(al.yaw) - np.mean(al.roll))
    flist.append(correlation(al.pitch,
                             al.roll,
                             np.mean(al.pitch),
                             np.mean(al.roll)))
    flist.append(np.mean(al.pitch) - np.mean(al.roll))
    flist.append(correlation(al.accx,
                             al.accy,
                             np.mean(al.accx),
                             np.mean(al.accy)))
    flist.append(np.mean(al.accx) - np.mean(al.accy))
    flist.append(correlation(al.accx,
                             al.accz,
                             np.mean(al.accx),
                             np.mean(al.accz)))
    flist.append(np.mean(al.accx) - np.mean(al.accz))
    flist.append(correlation(al.accy,
                             al.accz,
                             np.mean(al.accy),
                             np.mean(al.accz)))
    flist.append(np.mean(al.accy) - np.mean(al.accz))
    flist.append(correlation(al.rotx,
                             al.roty,
                             np.mean(al.rotx),
                             np.mean(al.roty)))
    flist.append(np.mean(al.rotx) - np.mean(al.roty))
    flist.append(correlation(al.rotx,
                             al.rotz,
                             np.mean(al.rotx),
                             np.mean(al.rotz)))
    flist.append(np.mean(al.rotx) - np.mean(al.rotz))
    flist.append(correlation(al.roty,
                             al.rotz,
                             np.mean(al.roty),
                             np.mean(al.rotz)))
    flist.append(np.mean(al.roty) - np.mean(al.rotz))
    return flist


def generate_features(x, cf: config.Config, include_absolute_features=True):
    """ Create a list of statistical features for a sequence of values
    corresponding to one type of sensor (e.g., acceleration, rotation, location).
    Features are split into two types:
     - "absolute": These tell information about the actual value of the sensors (e.g. mean, max, min, etc)
     - "relative": These tell information about the relations between sensor values (variance, skewness, etc)
    If include_absolute_features = False, don't include the "absolute" features.
    """
    flist = list()

    while len(x) > cf.samplesize:  # remove elements outside current window
        x = x[:cf.samplesize]

    # Set up values needed in both absolute and relative features:
    mean_value = np.mean(x)
    median_value = np.median(x)

    # Absolute features:
    if include_absolute_features:
        flist.append(max(x))
        flist.append(min(x))

        flist.append(np.sum(x))

        flist.append(mean_value)

        flist.append(median_value)

        newmap = [abs(number) for number in x]

        mav1 = np.mean(newmap)  # mean absolute value
        flist.append(mav1)

        mav2 = np.median(newmap)  # median absolute value
        flist.append(mav2)

        if cf.fftfeatures == 1:
            ceps, entropy, energy, msignal, vsignal = fft_features(x)
            flist.append(ceps)
            flist.append(entropy)
            flist.append(energy)
            flist.append(msignal)
            flist.append(vsignal)

        se = signal_energy(x)
        flist.append(se)

        lse = log_signal_energy(x)
        flist.append(lse)

        p = se / len(x)  # power
        flist.append(p)

    # Relative features:
    flist.append(np.var(x))

    std = np.std(x)
    flist.append(std)  # standard deviation

    flist.append(mean_absolute_deviation(x))
    flist.append(median_absolute_deviation(x))

    flist.append(zero_crossings(x, median_value))
    flist.append(mean_crossings(x, mean_value))

    m1, m2, m3, m4 = moments(x)
    flist.append(m1)
    flist.append(m2)
    flist.append(m3)
    flist.append(m4)

    flist.append(interquartile_range(x))

    if mean_value == 0:
        coefficient_of_variation = 0
    else:
        coefficient_of_variation = std / mean_value
    flist.append(coefficient_of_variation)

    flist.append(skewness(x, mean_value))

    k = kurtosis(x, mean_value, std)
    flist.append(k)

    if len(x) > 1:
        ac = autocorrelation(x, mean_value)
    else:
        ac = 0
    flist.append(ac)

    diff1 = list()
    diff2 = list()
    for i in range(len(x)):
        if i > 0:  # absolute differences between successive data points
            diff1 = np.append(diff1, np.absolute(x[i] - x[i - 1]))
            # absolute difference between each value and mean
        diff2 = np.append(diff2, np.absolute(x[i] - mean_value))
    flist.append(np.mean(diff1))
    flist.append(np.std(diff1))
    flist.append(np.mean(diff2))
    flist.append(np.std(diff2))

    tbpeaks = time_between_peaks(x)  # time between peaks
    flist.append(tbpeaks)

    return flist


def calculate_time_features(st, dt: datetime.datetime) -> Tuple[int, int, int, int, int]:
    """
    Calculate time features for one window of sensor data.
    """

    month = dt.month
    dayofweek = dt.weekday()
    hours = dt.hour
    minutes = (dt.hour * 60) + dt.minute
    seconds = (dt.hour * 3600) + (dt.minute * 60) + dt.second

    return month, dayofweek, hours, minutes, seconds


def calculate_space_features(st) -> Tuple[float, float, float, float]:
    """
    Calculate space features for one window of sensor data.
    """

    distance = math.sqrt(((st.maxlat - st.minlat) * (st.maxlat - st.minlat)) +
                         ((st.maxlong - st.minlong) * (st.maxlong - st.minlong)))
    hcr = heading_change_rate(st.course, distance)
    sr = stop_rate(st.latitude, st.longitude, distance)
    traj = trajectory(st.latitude, st.longitude)

    return distance, hcr, sr, traj


def create_point(st, dt, filename, person_stats, clusters):
    """
    Create a vector representing features for one window of sensor data.
    """
    xpoint = list()

    month, dayofweek, hours, minutes, seconds, distance, hcr, sr, pt_trajectory = \
        calculate_time_and_space_features(st, dt)

    for i in [st.yaw, st.pitch, st.roll, st.rotx, st.roty,
              st.rotz, st.accx, st.accy, st.accz, st.acctotal]:
        if st.conf.filter_data:
            i = utils.butter_lowpass_filter(i)

        xpoint.extend(generate_features(x=i, cf=st.conf))

    xpoint.extend(twod_features(st))

    if st.conf.local == 1:
        for i in [st.latitude, st.longitude, st.altitude]:
            # Only include absolute features if enabled in config:
            xpoint.extend(generate_features(x=i, cf=st.conf, include_absolute_features=st.conf.gen_gps_abs_stat_features))

        for i in [st.course, st.speed, st.hacc, st.vacc]:
            xpoint.extend(generate_features(x=i, cf=st.conf))

        xpoint += [distance, hcr, sr, pt_trajectory]

    xpoint += [month, dayofweek, hours, minutes, seconds]

    if st.conf.sfeatures == 1:
        xpoint += person.calculate_person_features(filename, st, person_stats, clusters)

    if st.conf.gpsfeatures == 1:
        if st.conf.locmodel == 1:
            newname = st.location.find_location(st.latitude[-1], st.longitude[-1])

            if newname is None:
                newname = st.location.label_loc(st, distance, hcr, sr,
                                                pt_trajectory, month, dayofweek, hours, minutes,
                                                seconds)
        else:
            place = st.location.generate_gps_features(np.mean(st.latitude),
                                                      np.mean(st.longitude))

            newname = st.location.map_location_name(place)

        xpoint.extend(st.location.generate_location_features(newname))

    return xpoint


def load_clusters(base_filename: str, cf: config.Config):
    """ Load the pre-trained person-specific cluster models.
    """
    clusters = list()
    for i in range(cf.num_hour_clusters):
        filename = os.path.join(cf.clusterpath, '{}.{}'.format(base_filename, i))
        clusters.append(joblib.load(filename))
    return clusters
