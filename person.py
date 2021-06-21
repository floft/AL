#!/usr/bin/python

# Contains functions to calculate person-specific features from sensor data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os
from collections import Counter
from multiprocessing import Process
from typing import List

import numpy as np

import config

from mobiledata import MobileData


def calculate_mean_distance(spanlat, spanlong, lat1, long1, lat2, long2):
    """ Calculate distance from person's overall mean location.
    """
    if spanlat == 0 or spanlong == 0:
        return 0.0
    else:
        d1 = (lat2 - lat1) * (lat2 - lat1)
        d2 = (long2 - long1) * (long2 - long1)
        span = np.sqrt(spanlat * spanlat + spanlong * spanlong)
        return np.sqrt(d1 + d2) / span


def calculate_distance(lat1, long1, lat2, long2):
    d1 = (lat2 - lat1) * (lat2 - lat1)
    d2 = (long2 - long1) * (long2 - long1)
    return np.sqrt(d1 + d2)


def close_loc(lat1, long1, lat2, long2):
    """ Determine if one location is within threshold distance to another.
    """
    distance = calculate_distance(lat1, long1, lat2, long2)
    if distance < 0.001:
        return True
    else:
        return False


def most_common(data):
    """ Return item that appears most often in list."
    """
    c = Counter(data)
    return c.most_common(1)[0][0]


def calculate_person_features(al, person_stats):
    """ Calculate person-specific features to include in feature vector.
    """
    results = list()
    spanlat = person_stats[-2]
    spanlong = person_stats[-1]
    meanlat = person_stats[-5]
    meanlong = person_stats[-4]

    results.append(calculate_mean_distance(spanlat, spanlong, meanlat, meanlong,
                                      np.mean(al.latitude), np.mean(al.longitude)))

    results.append(np.mean(al.latitude) - meanlat)
    results.append(np.mean(al.longitude) - meanlong)

    for i in (0, 9, 18, 27, 36):
        distances = np.zeros((3))
        size = min(al.conf.numseconds, len(al.latitude))
        lat1 = person_stats[i]
        long1 = person_stats[i+1]
        distances[0] = \
            calculate_distance(al.latitude[0], al.longitude[0], lat1, long1)
        for j in range(2, size):
            temp = calculate_distance(al.latitude[j], al.longitude[j], \
                                      lat1, long1)
            if temp < distances[0]:
                distances[0] = temp
        lat2 = person_stats[i+3]
        long2 = person_stats[i+4]
        distances[1] = \
            calculate_distance(al.latitude[0], al.longitude[0], lat2, long2)
        for j in range(2, size):
            temp = calculate_distance(al.latitude[j], al.longitude[j], \
                                      lat2, long2)
            if temp < distances[1]:
                distances[1] = temp
        lat3 = person_stats[i+6]
        long3 = person_stats[i+7]
        distances[2] = \
            calculate_distance(al.latitude[0], al.longitude[0], lat3, long3)
        for j in range(2, size):
            temp = calculate_distance(al.latitude[j], al.longitude[j], \
                                      lat3, long3)
            if temp < distances[2]:
                distances[2] = temp
        closest = np.argmin(distances)
        closest_distance = np.min(distances)
        if closest_distance < 0.01:
           results.append(closest)
        else:
           results.append(4)
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
            if event['latitude'] <= -90.0 or event['latitude'] >= 90.0 or event['latitude'] == 0.0:
                continue

            # At this point, assume all values are valid for this event and add to lists:
            latitude.append(event['latitude'])
            longitude.append(event['longitude'])
            altitude.append(event['altitude'])
            hour.append(event[cf.stamp_field_name].hour)

    return latitude, longitude, altitude, hour


def generate_person_stats(base_filename, cf: config.Config):
    """ Generate person-specific statistics from all sensor data for that
    person, labeled or unlabeled.
    """
    threshold = 1200   # 2 minutes at 10Hz sample rate

    latitude, longitude, altitude, hour = read_locations(base_filename, cf)

    n = len(hour)
    staypoints = []

    loc1x = latitude[0]
    loc1y = longitude[0]
    loc1z = altitude[0]

    count = 1
    stay = 1

    hour1 = hour[0]

    while count < n:
        loc2x = latitude[count]
        loc2y = longitude[count]
        loc2z = altitude[count]

        if close_loc(loc1x, loc1y, loc2x, loc2y):
            stay += 1
        else:
            if stay >= threshold:
                hour2 = hour[count]
                staypoints.append((loc1x, loc1y, loc1z, stay, hour1, hour2))

            loc1x = latitude[count]
            loc1y = longitude[count]
            loc1z = altitude[count]

            stay = 1

            hour1 = hour[count]

        count += 1

    if stay >= threshold:
        hour2 = hour[count-1]
        staypoints.append((loc1x, loc1y, loc1z, stay, hour1, hour2))

    stats = process_staypoints(staypoints)
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


def process_staypoints(staypoints):
    n = len(staypoints)

    sp0_24 = []
    sp0_6 = []
    sp6_12 = []
    sp12_18 = []
    sp18_24 = []
    new_sp = []

    for sp in staypoints:
        found = False

        for nsp in new_sp:
            if close_loc(sp[0], sp[1], nsp[0], nsp[1]):
                found = True

                new_sp.remove(nsp)
                temp = (nsp[0], nsp[1], nsp[2], nsp[3] + sp[3], nsp[4], nsp[5])
                new_sp.append(temp)

                break

        if not found:
            new_sp.append(sp)

    for sp in new_sp:
        if sp[4] in range(0, 6) or sp[4] in range(0, 6):
            sp0_6.append(sp)
        elif sp[4] in range(6, 12) or sp[4] in range(6, 12):
            sp6_12.append(sp)
        elif sp[4] in range(12, 18) or sp[4] in range(12, 18):
            sp12_18.append(sp)
        else:
            sp0_24.append(sp)

    mf = [staypoints[0], staypoints[0], staypoints[0]]
    mf = most_frequent_staypoints(staypoints, mf)

    stats = most_frequent_staypoints(sp0_6, mf)
    stats = np.append(stats, most_frequent_staypoints(sp6_12, mf))
    stats = np.append(stats, most_frequent_staypoints(sp12_18, mf))
    stats = np.append(stats, most_frequent_staypoints(sp18_24, mf))
    stats = np.append(stats, mf)
    return stats


def most_frequent_staypoints(sprange, mf):
    n = len(sprange)

    if n == 0:   # no staypoints
        return mf
    else:
        v1 = (sprange[0][0], sprange[0][1], sprange[0][2])
        c1 = sprange[0][3]

        if n == 1:   # one staypoints
            return [v1, v1, v1]
        else:
            if sprange[1][3] > sprange[0][3]:  # second staypoing more frequent
                v2 = v1
                c2 = c1
                v1 = (sprange[1][0], sprange[1][1], sprange[1][2])
                c1 = sprange[1][3]
            else:
                v2 = (sprange[1][0], sprange[1][1], sprange[1][2])
                c2 = sprange[1][3]

            if n == 2:   # two staypoints
                return [v1, v2, v2]
            else:
                if sprange[2][3] < sprange[1][3]:
                    v3 = (sprange[2][0], sprange[2][1], sprange[2][2])
                    c3 = sprange[2][3]
                elif sprange[2][3] < sprange[0][3]:
                    v3 = v2
                    c3 = c2
                    v2 = (sprange[2][0], sprange[2][1], sprange[2][2])
                    c2 = sprange[2][3]
                else:
                    v3 = v2
                    c3 = c2
                    v2 = v1
                    c2 = c1
                    v1 = (sprange[2][0], sprange[2][1], sprange[2][2])
                    c1 = sprange[2][3]

                for i in range(4, n):
                    if sprange[i][3] > c3:
                        if sprange[i][3] > c2:
                            if sprange[i][3] > c1:
                                v3 = v2
                                c3 = c2
                                v2 = v1
                                c2 = c1
                                v1 = (sprange[i][0],sprange[i][1],sprange[i][2])
                                c1 = sprange[i][3]
                            else:
                                v3 = v2
                                c3 = c2
                                v2 = (sprange[i][0],sprange[i][1],sprange[i][2])
                                c2 = sprange[i][3]
                        else:
                            v3 = (sprange[i][0], sprange[i][1], sprange[i][2])
                            c3 = sprange[i][3]
                            
    return [v1, v2, v3]


def main(base_filename, cf: config.Config):
    stats = generate_person_stats(base_filename, cf)
    personfile = os.path.join(cf.datapath, base_filename + '.person')
    np.savetxt(personfile, stats, delimiter=',')
    return


def parallel_generate_person_stats(base_filename: str, cf: config.Config):
    """
    Generate person stats for the given file base filename and config. Intended to be used in
    parallel processes.

    Parameters
    ----------
    base_filename : str
        The base name for the file to parse person features for
    cf : config.Config
        The configuration for the processing
    """

    print(f'Creating person features for {base_filename}')

    main(base_filename, cf)

    print(f'Finished person features for {base_filename}')


def generate_multiple_person_stats(files: List[str], cf: config.Config):
    """
    Generate person features for multiple files in parallel using multiprocessing.

    Parameters
    ----------
    files : List[str]
        The list of base filenames to parse features for
    cf : config.Config
        The config object for configuring parsing
    """

    # Verify that the files all exist:
    for base_filename in files:
        full_filename = os.path.join(conf.datapath, base_filename + conf.extension)

        if not os.path.isfile(full_filename):
            msg = f"{full_filename} does not exist"
            raise ValueError(msg)

    # Now run the multiprocessing:
    # Create list of processes:
    stat_gen_processes = list()

    # Set up the stat generation process objects:
    for base_filename in files:
        stat_gen_processes.append(Process(target=parallel_generate_person_stats,
                                          args=(base_filename, cf)))

    # Start the processes:
    for i in range(len(stat_gen_processes)):
        stat_gen_processes[i].start()

    # Join the processes back when done:
    for i in range(len(stat_gen_processes)):
        stat_gen_processes[i].join()


if __name__ == "__main__":
    conf = config.Config(description='')
    conf.set_parameters()

    generate_multiple_person_stats(conf.files, conf)
