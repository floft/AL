#!/usr/bin/python

# Looks up location type information for each unique GPS coordinate in the
# <gps_coord_file>, starting from line <start_line>, and writes the information
# to the locations file. If locations file already exists, then it first reads
# all the previous location information. This file is then over-written with
# both the old and new information.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2017. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.
import math
import os
from argparse import ArgumentParser
from operator import itemgetter
from typing import Optional, List

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


def gps_find_location(gps_locations: List[List], lat: float, long: float) -> Optional[str]:
    """
    Look in the list of gps_locations for one that is "close enough" to the provided lat,long
    coordinate, and return its type (3rd value of tuple) if found.

    Parameters
    ----------
    gps_locations: List[List]
        A list of location tuples/lists (lat, long, type, ...)
    lat : float
        The latitude to search for
    long : float
        The longitude to search for

    Returns
    -------
    Optional[str]
        If a "close enough" location is found, return its type (3rd value of tuple). Otherwise,
        return None.
    """

    threshold = 0.0005

    for loc_tuple in gps_locations:
        tlat = loc_tuple[0]
        tlong = loc_tuple[1]

        dist = math.sqrt(((tlat - lat) * (tlat - lat)) + ((tlong - long) * (tlong - long)))

        if dist < threshold:
            return loc_tuple[2]

    return None


# The following is_type() functions are used to determine if a certain reverse-geocoded location
# fits into a certain core location type that we use. This is mostly used in `print_features()`.
# In the future we may want to look at integrating this with the loc.translate options
def is_house(loc_type, loc_class):
    if loc_type == 'house' or loc_class == 'house' or \
            loc_type == 'hamlet' or loc_class == 'hamlet' or \
            loc_type == 'hotel' or loc_class == 'hotel' or \
            loc_type == 'motel' or loc_class == 'motel' or \
            loc_type == 'camp_site' or loc_type == 'neighborhood' or \
            loc_type == 'neighbourhood' or loc_type == 'retirement_home' or \
            loc_type == 'residential' or loc_class == 'residential' or \
            loc_type == 'private_residence' or loc_type == 'suburb' or \
            loc_type == 'nursing_home' or loc_type == 'neighbourhood':
        return True
    else:
        return False


def is_restaurant(loc_type, loc_class):
    if loc_type == 'bar' or loc_class == 'bar' or \
            loc_type == 'restaurant' or loc_class == 'restaurant' or \
            loc_type == 'bakery' or loc_class == 'bakery' or \
            loc_type == 'bbq' or loc_class == 'bbq' or \
            loc_type == 'brewery' or loc_class == 'brewery' or \
            loc_type == 'alcohol' or loc_class == 'alcohol' or \
            loc_type == 'cafe' or loc_type == 'coffee' or \
            loc_type == 'pub' or loc_class == 'pub' or \
            loc_type == 'fast_food' or loc_class == 'fast_food' or \
            loc_type == 'biergarten' or loc_type == 'confectionery' or \
            loc_type == 'food' or loc_type == 'food_court' or \
            loc_type == 'seafood' or loc_type == 'seafood' or \
            loc_type == 'deli;convenience' or loc_type == 'nightclub':
        return True
    else:
        return False


def is_road(loc_type, loc_class):
    if loc_type == 'highway' or loc_class == 'highway' or \
            loc_type == 'motorway' or loc_class == 'motorway' or \
            loc_type == 'motorway_junction' or loc_class == 'motorway_junction' or \
            loc_type == 'motorway_link' or loc_class == 'motorway_link' or \
            loc_type == 'parking' or loc_class == 'parking' or \
            loc_type == 'parking_entrance' or loc_type == 'parking_space' or \
            loc_type == 'bus_stop' or loc_class == 'bus_stop' or \
            loc_type == 'ferry_terminal' or loc_type == 'motorcycle' or \
            loc_type == 'cycleway' or loc_class == 'cycleway' or \
            loc_type == 'footway' or loc_class == 'footway' or \
            loc_type == 'fuel' or loc_class == 'fuel' or \
            loc_type == 'trunk' or loc_class == 'trunk' or \
            loc_type == 'road' or loc_class == 'road' or \
            loc_type == 'pedestrian' or loc_type == 'rest_area' or \
            loc_type == 'terminal' or loc_class == 'terminal' or \
            loc_class == 'railway' or loc_type == 'water' or \
            loc_type == 'hangar' or loc_type == 'taxi_way' or \
            loc_type == 'track' or loc_type == 'primary' or \
            loc_type == 'secondary' or loc_type == 'tertiary' or \
            loc_type == 'bus_station' or loc_type == 'bridge':
        return True
    else:
        return False


def is_store(loc_type, loc_class):
    if loc_type == 'bank' or loc_class == 'bank' or \
            loc_type == 'bureau_de_change' or loc_class == 'bureau_de_change' or \
            loc_type == 'gold_exchange' or loc_type == 'watches' or \
            loc_type == 'bicycle_rental' or loc_class == 'bicycle_rental' or \
            loc_type == 'bicycle_repair_station' or loc_class == 'bicycle_repair_station' or \
            loc_type == 'boutique' or loc_class == 'boutique' or \
            loc_type == 'art' or loc_class == 'art' or loc_type == 'gallery' or \
            loc_type == 'art_class' or loc_class == 'art_class' or \
            loc_type == 'auto_parts' or loc_class == 'auto_parts' or \
            loc_type == 'beauty' or loc_class == 'beauty' or \
            loc_type == 'beauty_supply' or loc_class == 'beauty_supply' or \
            loc_type == 'books' or loc_class == 'books' or \
            loc_type == 'furniture' or loc_class == 'car_wash' or \
            loc_type == 'shop' or loc_class == 'shop' or \
            loc_type == 'supermarket' or loc_class == 'supermarket' or \
            loc_type == 'greengrocer' or loc_type == 'ice_cream' or \
            loc_type == 'marketplace' or loc_type == 'video' or \
            loc_type == 'clothes' or loc_class == 'clothes' or \
            loc_type == 'insurance' or loc_class == 'insurance' or \
            loc_type == 'interior_decoration' or \
            loc_type == 'marketplace' or loc_class == 'marketplace' or \
            loc_type == 'atm' or loc_type == 'insurance' or \
            loc_type == 'pharmacy' or loc_type == 'nutrition_supplements' or \
            loc_type == 'department_store' or loc_type == 'store' or \
            loc_type == 'electronics' or loc_type == 'garden_centre' or \
            loc_type == 'jewelry' or loc_type == 'retail' or loc_type == 'mall' or \
            loc_type == 'toys' or loc_type == 'tuxedo' or loc_type == 'soap' or \
            loc_type == 'marketplace' or loc_type == 'variety_store' or \
            loc_type == 'doityouself':
        return True
    else:
        return False


def is_work(loc_type, loc_class):
    if loc_type == 'office' or loc_class == 'office' or \
            loc_type == 'school' or loc_class == 'school' or \
            loc_type == 'yes' or loc_class == 'yes' or \
            loc_type == 'accountant' or loc_class == 'accountant' or \
            loc_type == 'administrative' or loc_class == 'administrative' or \
            loc_type == 'government' or loc_type == 'lawyer' or \
            loc_type == 'public_building' or loc_class == 'building' or \
            loc_type == 'company' or loc_class == 'public_building' or \
            loc_type == 'kindergarten' or loc_type == 'university' or \
            loc_type == 'conference_center' or loc_type == 'college':
        return True
    else:
        return False


def is_attraction(loc_type, loc_class):
    if loc_type == 'golf_course' or loc_class == 'golf_course' or \
            loc_type == 'aerodrome' or loc_class == 'aerodrome' or \
            loc_type == 'attraction' or loc_class == 'attraction' or \
            loc_type == 'beach' or loc_class == 'beach' or \
            loc_type == 'garden' or loc_class == 'leisure' or \
            loc_type == 'tourism' or loc_class == 'tourism' or \
            loc_type == 'museum' or loc_class == 'museum' or \
            loc_type == 'theatre' or loc_class == 'theatre' or \
            loc_type == 'swimming_area' or loc_type == 'swimming_pool' or \
            loc_type == 'casino' or loc_type == 'cinema' or \
            loc_type == 'park' or loc_class == 'park' or \
            loc_type == 'lifeguard_tower' or loc_type == 'nature_reserve' or \
            loc_type == 'picnic_site' or loc_type == 'playground' or \
            loc_type == 'boat' or loc_class == 'boat' or \
            loc_type == 'river' or loc_class == 'river' or \
            loc_type == 'social_facility' or loc_class == 'social_facility' or \
            loc_type == 'sports_centre' or loc_type == 'stadium' or \
            loc_type == 'bench' or loc_class == 'bench':
        return True
    else:
        return False


def is_service(loc_type, loc_class):
    if loc_type == 'place_of_worship' or loc_class == 'place_of_worship' or \
            loc_type == 'fire_station' or loc_class == 'fire_station' or \
            loc_type == 'ranger_station' or loc_class == 'ranger_station' or \
            loc_type == 'fitness_centre' or loc_type == 'florist' or \
            loc_type == 'atm' or loc_class == 'atm' or \
            loc_type == 'townhall' or loc_class == 'townhall' or \
            loc_type == 'aeroway' or loc_class == 'aeroway' or \
            loc_type == 'car_wash' or loc_type == 'service' or loc_class == 'service' or \
            loc_type == 'hospital' or loc_class == 'hospital' or \
            loc_type == 'caravan_site' or loc_type == 'caterer' or \
            loc_type == 'clinic' or \
            loc_type == 'community_centre' or loc_type == 'artwork' or \
            loc_type == 'dentist' or loc_type == 'amenity' or \
            loc_class == 'historic' or loc_type == 'toilets' or \
            loc_type == 'post_box' or loc_class == 'emergency' or \
            loc_type == 'emissions_testing' or loc_type == 'library' or \
            loc_type == 'doctor' or loc_type == 'doctors' or loc_type == 'clinic' or \
            loc_type == 'dry_cleaning' or loc_type == 'optician' or \
            loc_type == 'doctors' or loc_type == 'shelter' or \
            loc_type == 'post_office' or loc_type == 'post_box' or \
            loc_class == 'landuse' or loc_type == 'car_rental' or \
            loc_type == 'car_repair' or loc_type == 'charging_station' or \
            loc_class == 'natural' or loc_type == 'books' or \
            loc_type == 'police' or loc_type == 'vending_machine' or \
            loc_type == 'veterinary' or loc_type == 'charging_station' or \
            loc_type == 'childcare' or loc_type == 'gym' or \
            loc_type == 'auto_repair' or \
            loc_type == 'tanning' or loc_type == 'car_sales' or \
            loc_type == 'car_sales' or loc_type == 'townhall' or \
            loc_type == 'compressed_air' or loc_type == 'chiropractor' or \
            loc_type == 'recycling' or loc_type == 'tutoring' or \
            loc_type == 'employment_agency' or loc_type == 'estate_agent' or \
            loc_type == 'realtor' or loc_class == 'realtor' or \
            loc_type == 'hunting_stand':
        return True
    else:
        return False


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

    print_loc_features = True  # print high-level features
    count = 0

    with open(in_file, 'r') as input_file:
        for line in input_file:
            if start <= count <= end:
                location = str(str(line).strip()).split(' ', 2)

                print('location ', location, 'count', count)

                loc = gps_find_location(location[0], location[1])  # look in file

                if loc is None:
                    address = get_address(location)

                    try:
                        description = geolocator.geocode(address, timeout=None)
                        print('description', description)

                        if address == 'None' or description == 'None' or description is None:
                            print_features(location[0], location[1], 'other', 'other')
                        else:
                            raw = description.raw

                            if print_loc_features:
                                print_features(location[0], location[1], raw['type'], raw['class'])
                            else:
                                print(raw['type'], ' ', raw['class'])
                    except:
                        print('Error, geocode failed')

                        continue  # don't add to our count

            count += 1

            # Stop when we reach end, to avoid looping through unnecessary lines:
            if count > end:
                break

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
