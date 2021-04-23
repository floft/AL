#!/usr/bin/python

# Contains functions to calculate person-specific features from sensor data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os
from collections import Counter

import joblib
import numpy as np

import config
import kmeans
import warnings

from mobiledata import MobileData


def calculate_distance(spanlat, spanlong, lat1, long1, lat2, long2):
    """ Calculate distance from person's overall mean location.
    """
    if spanlat == 0 or spanlong == 0:
        return 0.0
    else:
        d1 = (lat2 - lat1) * (lat2 - lat1)
        d2 = (long2 - long1) * (long2 - long1)
        return np.sqrt(d1 + d2) / (spanlat + spanlong)


def close_loc(lat1, long1, alt1, lat2, long2, alt2):
    """ Determine if one location is within threshold distance to another.
    """
    d1 = (lat2 - lat1) * (lat2 - lat1)
    d2 = (long2 - long1) * (long2 - long1)
    d3 = (alt2 - alt1) * (alt2 - alt1)
    distance = np.sqrt(d1 + d2 + d3)
    if distance < 3.0:
        return True
    else:
        return False


def most_common(data):
    """ Return item that appears most often in list."
    """
    c = Counter(data)
    return c.most_common(1)[0][0]


def calculate_person_features(feat_infile, al, person_stats, oc_clusters):
    """ Calculate person-specific features to include in feature vector.
    """
    results = list()
    spanlat = person_stats[-2]
    spanlong = person_stats[-1]
    meanlat = person_stats[-5]
    meanlong = person_stats[-4]

    results.append(calculate_distance(spanlat, spanlong, meanlat, meanlong,
                                      np.mean(al.latitude), np.mean(al.longitude)))

    results.append(np.mean(al.latitude) - meanlat)
    results.append(np.mean(al.longitude) - meanlong)

    for i in range(al.conf.num_hour_clusters):
        array = list()
        size = min(al.conf.numseconds, len(al.latitude))
        for j in range(size):
            array.append((al.latitude[j], al.longitude[j], al.altitude[j]))

        km = kmeans.KMeans()
        labels = km.sorted_kmeans_predict(array, oc_clusters[i])
        results.append(most_common(labels))
    return results


def read_locations(base_filename: str, cf: config.Config):
    """ Read and store valid locations from input sensor file.
    """

    latitude = list()
    longitude = list()
    altitude = list()
    hour = list()

    loc_infile = os.path.join(cf.datapath, base_filename + cf.extension)

    # Read all events from the CSV file:
    with MobileData(loc_infile, 'r') as in_data:
        for event in in_data.rows_dict:
            # Skip this event if lat/long/alt values are missing:
            if event['latitude'] is None or event['longitude'] is None or event['altitude'] is None:
                continue

            # Skip if the latitude value is invalid (outside range):
            if event['latitude'] <= -90.0 or event['latitude'] >= 90.0:
                continue

            # At this point, assume all values are valid for this event and add to lists:
            latitude.append(event['latitude'])
            longitude.append(event['longitude'])
            altitude.append(event['altitude'])
            hour.append(event[cf.stamp_field_name].hour)

    return latitude, longitude, altitude, hour


def hour_cluster(base_filename, latitude, longitude, altitude,
                 hour, n, begin, end, numclusters, cluster_num, cf: config.Config):
    """ Cluster locations that appear within specified time window.
    """
    locs = list()
    for i in range(n):  # consider points with times between begin and end
        if ((end < begin) and (hour[i] >= begin or hour[i] <= end)) or \
                (end > begin and begin <= hour[i] <= end):
            new = np.zeros(3)
            new[0] = latitude[i]
            new[1] = longitude[i]
            new[2] = altitude[i]
            locs.append(new)
    n = len(locs)
    for i in range(n, numclusters):  # make sure enough points to create clusters
        new = np.zeros(3)
        if n == 0:
            locs.append(new)
        else:
            new[0] = latitude[0]
            new[1] = longitude[0]
            new[2] = altitude[0]
            locs.append(new)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        km = kmeans.KMeans()
        km.sorted_kmeans_fit(locs, numclusters)

    filename = os.path.join(cf.clusterpath, '{}.{}'.format(base_filename, cluster_num))
    joblib.dump(km.centers, filename)
    return np.ndarray.flatten(km.centers)


def generate_person_stats(base_filename, cf: config.Config):
    """ Generate person-specific statistics from all sensor data for that person,
    labeled or unlabeled. Statistics include:
        1. centers of location clusters at times throughout the day
        2. top-n most-frequent locations
        3. center of visited locations
        4. span of visited locations
    """
    latitude, longitude, altitude, hour = read_locations(base_filename, cf)
    n = len(hour)
    stats = hour_cluster(base_filename, latitude, longitude, altitude, hour, n, 0, 6, 3, 0, cf)
    cc = hour_cluster(base_filename, latitude, longitude, altitude, hour, n, 6, 12, 3, 1, cf)
    stats = np.append(stats, cc)
    cc = hour_cluster(base_filename, latitude, longitude, altitude, hour, n, 12, 18, 3, 2, cf)
    stats = np.append(stats, cc)
    cc = hour_cluster(base_filename, latitude, longitude, altitude, hour, n, 18, 24, 3, 3, cf)
    stats = np.append(stats, cc)
    cc = hour_cluster(base_filename, latitude, longitude, altitude, hour, n, 0, 24, 3, 4, cf)
    stats = np.append(stats, cc)
    stats = np.append(stats, np.mean(latitude))
    stats = np.append(stats, np.mean(longitude))
    stats = np.append(stats, np.mean(altitude))
    minvalue = np.min(latitude)
    maxvalue = np.max(latitude)
    stats = np.append(stats, maxvalue - minvalue)
    minvalue = np.min(longitude)
    maxvalue = np.max(longitude)
    stats = np.append(stats, maxvalue - minvalue)
    return [stats]


def main(base_filename, cf: config.Config):
    stats = generate_person_stats(base_filename, cf)
    personfile = os.path.join(cf.datapath, base_filename + '.person')
    np.savetxt(personfile, stats, delimiter=',')
    return


if __name__ == "__main__":
    conf = config.Config(description='')
    conf.set_parameters()

    for infile in conf.files:
        full_infile = os.path.join(conf.datapath, infile + conf.extension)
        if not os.path.isfile(full_infile):
            print(full_infile, "does not exist")
        else:
            main(base_filename=infile,
                 cf=conf)
