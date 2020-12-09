#!/usr/bin/python

# Looks up location type information for each unique GPS coordinate in the
# <gps_coord_file>, starting from line <start_line>, and writes the information
# to the locations file. If locations file already exists, then it first reads
# all the previous location information. This file is then over-written with
# both the old and new information.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2017. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os
from operator import itemgetter

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="AL")  # open street map server


def get_address(loc):
    """ Use the open street map to retrieve an address corresponding to a
    latitude, longitude location.
    """
    lat = loc[0]
    long = loc[1]
    location = str(str(lat) + ", " + str(long))
    try:
        location = geolocator.reverse(location)
        astr = str(location.raw)
        print('found', astr)
        return location.address
    except:
        return 'None'


def gps_read_locations(lfile):
    """ Read a set of locations (latitude and longitude) with
    corresponding location types (three types of varying abstraction) from a file.
    """
    gps_locations = list()

    if os.path.isfile(lfile):
        with open(lfile, "r") as file:
            for line in file:
                x = str(str(line).strip()).split(' ', 4)
                gps_tuple = list()
                gps_tuple.append(float(x[0]))
                gps_tuple.append(float(x[1]))
                gps_tuple.append(x[2])
                gps_tuple.append(x[3])
                gps_tuple.append(x[4])
                gps_locations.append(gps_tuple)
    return


def update_locations(locations, locationsfile):
    locations = sorted(locations, key=itemgetter(0))
    unique_locations = list()
    output = open(locationsfile, "w")
    for location in locations:
        if len(location) > 4:
            if location not in unique_locations:
                output.write(str(location[0]) + ' ')
                output.write(str(location[1]) + ' ')
                output.write(location[2] + ' ')
                output.write(location[3] + ' ')
                output.write(location[4] + '\n')
        unique_locations.append(location)
    output.close()
    return


def get_location_type(location, locationsfile):
    description = 'None'

    address = get_address(location)
    if address == 'None' or address is None:
        return 'Other'
    description = geolocator.geocode(address, timeout=None)
    print('description', description)
    if description == 'None' or description is None:
        return 'Other'
    else:
        raw = description.raw
        loc_type = raw['type']
        if loc_type == 'other' or loc_type == 'Other':
            return loc_type
        else:
            gps_read_locations(locationsfile)
            return loc_type
