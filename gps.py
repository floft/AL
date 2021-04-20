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
from argparse import ArgumentParser
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

    return gps_locations


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
        return 'other'
    description = geolocator.geocode(address, timeout=None)
    print('description', description)
    if description == 'None' or description is None:
        return 'other'
    else:
        raw = description.raw
        loc_type = raw['type']

        return loc_type


def geocode_lat_longs(in_file: str, locations_file: str, start: int, end: int):
    """
    Reverse-geocode the lat/long pairs in the input file using Nominatim, and write the geocoded
    tuples to the output locations file. Only geocode those lines in the input between start and
    end.

    Parameters
    ----------
    in_file : str
        Name of the input lat/long pair file to read from (each row should be "lat long" values
    locations_file : str
        The locations file name to write the locations out to
    start : int
        Line number of in the input to start at
    end : int
        Line number of the input to end at
    """

    # Read the existing locations file:
    gps_locs = gps_read_locations(locations_file)

    print_features = True  # print high-level features
    count = 0

    with open(in_file, 'r') as input_file:
        for line in input_file:
            if count >= start and count <= end:
                location = str(str(line).strip()).split(' ', 2)

                print('location ', location, 'count', count)

                loc = gps_find_location(location[0], location[1])  # look in file

                if loc is None:
                    address = get_address(location)

                    try:
                        description = geolocator.geocode(address, timeout=None)
                        print('description', description)

                        if address == 'None' or description == 'None' or description == None:
                            print_features(location[0], location[1], 'other', 'other')
                        else:
                            raw = description.raw

                            if print_features:
                                print_features(location[0], location[1], raw['type'], raw['class'])
                            else:
                                print(raw['type'], ' ', raw['class'])
                    except:
                        print('Error, geocode failed')

                        continue  # don't add to our count

            count += 1

    update_locations(gps_locs, locations_file)


if __name__ == '__main__':
    """
    Run script to reverse-geocode multiple locations in lat/long file if called as a script.
    """

    parser = ArgumentParser(description="Reverse-geocode locations in a lat/long file")

    parser.add_argument('in_file', type=str, default='latlong',
                        help="Input file (default %(default)s)")
    parser.add_argument('start', type=int, default=0,
                        help="Start position in the file (default %(default)s)")
    parser.add_argument('locations_file', type=str, default='locations',
                        help="Output file to write geocoded locations to (default %(default)s)")
    parser.add_argument('end', type=int, default=27500000,
                        help="End position in the file (default %(default)s)")

    args = parser.parse_args()

    # Geocode the lat/long entries in the input file and write to location file:
    geocode_lat_longs(args.in_file, args.start, args.location_file, args.end)
