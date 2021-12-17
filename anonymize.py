#!/usr/bin/env python3
"""
Anonymize CSV files

- Replace location data with location type
- Set the timestamps of each file to start at January 1, 2000 (+ a few days
  to get the day of the week to match)

Usage:
    mkdir anonymized
    python3 anonymize.py --dataset=younger --output_path=anonymized watch*.instances.csv
    python3 anonymize.py --dataset=older --output_path=anonymized sttr*.instances.csv
"""
import os
import math

from absl import app
from absl import flags
from datetime import datetime, timedelta
from typing import OrderedDict

from pool import run_job_pool
from mobiledata import MobileData

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "Number of jobs to use for processing files (0 = number of cores)")
flags.DEFINE_enum("dataset", None, ["younger", "older"], "Which dataset the files are from")
flags.DEFINE_string("output_path", None, "If not empty, then override the output path - all files will be put in this directory")
flags.DEFINE_string("location_file", "locations", "Mapping from GPS lat/lon to location type, which can be generated via gps.py")
flags.DEFINE_string("location_translation_file", "loc.translate", "Translation file for location types")
flags.DEFINE_string("label_translation_file", "act.translate", "Translation file for activity labels")


def get_label(
    row,
    label_map,
    class_labels,
    translate_labels,
):
    """ Get mapped label, and if we want to skip it, then return None """
    label = row["user_activity_label"]

    if translate_labels:
        label = label_map[label]

    if label == "Other" or label == "Ignore":
        label = None
    else:
        label = class_labels.index(label)

    return label


def get_start_timestamp(
    filename,
    label_map,
    class_labels,
    translate_labels,
):
    """ Get the start timestamp from the file """
    with MobileData(filename, "r") as data:
        start_ts = None

        for row in data.rows_dict:
            label = get_label(row, label_map, class_labels, translate_labels)

            if label is None:
                continue

            # Date/time features
            ts = row["stamp"]

            # Save the first one
            if start_ts is None:
                start_ts = ts
                break

    return start_ts


def get_location_type(
    gps_locations,
    gps_cache,
    lat,
    lon,
):
    """
    Find a "close enough" location to the lat/lon provided in the list of
    GPS locations. If there isn't one, return None.

    Based on: gps.py
    """
    # Return cached value if available
    location_key = (lat, lon)

    if location_key in gps_cache:
        return gps_cache[location_key]

    # If not in the cache, check our list of locations
    threshold = 0.0005
    location_type = None

    for tlat, tlon, tlocation_type in gps_locations:
        dist = math.sqrt(((tlat - lat) * (tlat - lat)) + ((tlon - lon) * (tlon - lon)))

        if dist < threshold:
            location_type = tlocation_type

    # Save to cache, even if None (here we do not support loading the
    # reverse geocoding - see gps.py to compute those)
    gps_cache[location_key] = location_type

    return location_type


def get_locations(
    location_filename,
    location_translation_filename=None,
):
    """
    List of (lat, lon, location_types)

    If translation file is provided, map the reverse geocoded location
    type to possibly other names, e.g., to reduce the number of location
    types.

    Get all the locations needed in the files with (assuming lat/lon are in
    columns 11 and 12):
        cat watch*.instances.csv sttr*.instances.csv | grep -v stamp,yaw | \
            grep -v dt,f,f | cut -d',' -f11,12 | tr ',' ' ' | \
            grep -v -e '^[[:space:]]*$' | sort -u > locations.txt

    Then, use the gps.py to do the reverse lookups:
        python3 gps.py locations.txt 0 locations
    """
    # Load location type translation first
    location_translation = None

    if location_translation_filename is not None:
        location_translation = get_translation(location_translation_filename)

    # Create the (lat,lon,location_type) list
    locations = []

    with open(location_filename, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            assert len(parts) >= 3, \
                "locations file not in the right format: lat lon str str str"

            # Get the key and location type
            lat = float(parts[0])
            lon = float(parts[1])
            location_type = parts[2]

            # Map location type if needed
            if (
                location_translation is not None
                and location_type in location_translation
            ):
                location_type = location_translation[location_type]

            locations.append((lat, lon, location_type))

    return locations


def get_translation(filename):
    """ Translate location names, labels, etc.

    Format for the file (one per line) - we replace str1 with str2:
        str1 str2
    """
    translation = {}

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            assert len(parts) == 2, \
                "translation not in the right format: str str"

            # Replace str1 with str2
            str1 = parts[0]
            str2 = parts[1]
            translation[str1] = str2

    return translation


def process_file(
    input_filename,
    output_filename,
    locations,
    label_map,
    class_labels,
    translate_labels,
    verbose=False,
):
    if verbose:
        print(input_filename, "->", output_filename)

    rows = []

    location_types = [
        "other",
        "house",
        "road",
        "service",
        "work",
        "attraction",
    ]
    location_types_capitalized = [label.title() for label in location_types]
    class_labels_capitalized = [label.title() for label in class_labels]

    gps_cache = {}
    loc_already_printed = {}

    # Get start timestamp so we can set all data to start at January 1, 2000.
    # However, remove the minutes/seconds/etc. since we want to keep the time,
    # just not the absolute year/month/day.
    start_date = get_start_timestamp(
        input_filename, label_map, class_labels, translate_labels
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    reference_date = datetime(2000, 1, 1)

    # However, we need to make sure the day of week matches, so adjust
    # accordingly
    start_date_weekday = start_date.weekday()
    reference_date_weekday = reference_date.weekday()
    day_shift = start_date_weekday - reference_date_weekday

    if day_shift < 0:
        day_shift += 7
    assert day_shift >= 0

    adjusted_reference_date = reference_date + timedelta(days=day_shift)

    # Read each row, anonymize, and write
    with (
        MobileData(input_filename, "r") as data_input,
        MobileData(output_filename, "w") as data_output
    ):
        fields = OrderedDict({
            "stamp": "dt",
            "yaw": "f",
            "pitch": "f",
            "roll": "f",
            "rotation_rate_x": "f",
            "rotation_rate_y": "f",
            "rotation_rate_z": "f",
            "user_acceleration_x": "f",
            "user_acceleration_y": "f",
            "user_acceleration_z": "f",
            "altitude": "f",
            "speed": "f",
            "location_type": "s",
            "user_activity_label": "s",
        })
        data_output.set_fields(fields)
        data_output.write_headers()

        for row in data_input.rows_dict:
            #
            # Read and anonymize each row
            #
            label = get_label(row, label_map, class_labels, translate_labels)

            if label is None:
                continue

            ts = row["stamp"]

            # Anonymize timestamp -> set to start January 1, 2000
            new_ts = ts - start_date + adjusted_reference_date

            # But, make sure our computed date/time features are the same
            day_of_week = ts.weekday()  # 0 = Monday, ..., 6 = Sunday
            new_day_of_week = new_ts.weekday()
            hour_of_day = ts.hour  # 24-hour time
            new_hour_of_day = new_ts.hour
            midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            minutes_past_midnight = (ts - midnight).total_seconds() / 60
            new_midnight = new_ts.replace(hour=0, minute=0, second=0, microsecond=0)
            new_minutes_past_midnight = (new_ts - new_midnight).total_seconds() / 60
            assert day_of_week == new_day_of_week, \
                "day_of_week differs " + str(day_of_week) + " vs. " + str(new_day_of_week) + \
                ", " + str(ts) + " vs. " + str(new_ts)
            assert hour_of_day == new_hour_of_day, "hour_of_day differs"
            assert minutes_past_midnight == new_minutes_past_midnight, \
                "minutes past midnight differs"

            # Anonymize location - reverse geo-lookup -> string location type
            lat = row['latitude']
            lon = row['longitude']

            if lat is not None and lon is not None:
                location_type_str = get_location_type(locations, gps_cache, lat, lon)

                if location_type_str is not None:
                    location_type = location_types.index(location_type_str)
                else:
                    # Warn about missing locations, but only once per
                    # location -- otherwise stdout is spammed with tons
                    # of duplicate lines
                    key = (lat, lon)
                    if key not in loc_already_printed:
                        print("Warning: location not found", lat, lon, "(setting to 'other')")
                        loc_already_printed[key] = True

                    location_type = 0
            else:
                # No GPS lat/lon in the file
                location_type = 0

            # Capitalize for consistency with the label name strings
            location_type = location_types_capitalized[location_type]

            # Map label back to string and make sure it's capitalized
            label = class_labels_capitalized[label]

            assert location_type in location_types_capitalized
            assert label in class_labels_capitalized

            #
            # Write the anonymized row
            #
            data_output.write_row_dict({
                "stamp": new_ts,
                "yaw": row["yaw"],
                "pitch": row["pitch"],
                "roll": row["roll"],
                "rotation_rate_x": row["rotation_rate_x"],
                "rotation_rate_y": row["rotation_rate_y"],
                "rotation_rate_z": row["rotation_rate_z"],
                "user_acceleration_x": row["user_acceleration_x"],
                "user_acceleration_y": row["user_acceleration_y"],
                "user_acceleration_z": row["user_acceleration_z"],
                "altitude": row["altitude"],
                "speed": row["speed"],
                "location_type": location_type,
                "user_activity_label": label,
            })


def filename_renames(
    output_filenames,
    filename_rename,
    override_output_path=None,
):
    """
    Make a set of replacements to the filenames passed in via output_filenames,
    but not to the paths.

    For example, set filename_rename=[("a", "b")] and it will replace "a" with
    "b" in the filename, but not the path.
    """
    new_output_filenames = []

    for output_filename in output_filenames:
        path, fn = os.path.split(output_filename)

        for rename in filename_rename:
            fn = fn.replace(*rename)

        if override_output_path is not None:
            path = override_output_path

        new_output_filenames.append(os.path.join(path, fn))

    return new_output_filenames


def main(argv):
    # Input/output filenames
    filenames = argv[1:]

    if len(filenames) == 0:
        print("Error: No CSV filenames specified")
        exit()

    output_filenames = [f.replace(".csv", "").replace(".instances", "") + ".csv" for f in filenames]

    # Dataset-specific changes
    if FLAGS.dataset == "younger":
        class_labels = [
            "Cook",
            "Eat",
            "Exercise",
            "Hygiene",
            "Travel",
            "Work",
            # "Other",
        ]
        filename_rename = [("watch", "younger_adults_")]
        translate_labels = False
    elif FLAGS.dataset == "older":
        class_labels = [
            "Mealtime",
            "Errands",
            "Exercise",
            "Hobby",
            "Hygiene",
            "Relax",
            "Sleep",
            "Travel",
            "Work",
            # "Ignore",
        ]
        filename_rename = [("sttr", "older_adults_")]
        translate_labels = True
    else:
        raise NotImplementedError("Unknown --dataset option")

    output_filenames = filename_renames(output_filenames, filename_rename,
        override_output_path=FLAGS.output_path)

    # Translation
    locations = get_locations(
        FLAGS.location_file,
        FLAGS.location_translation_file,
    )

    if translate_labels:
        label_map = get_translation(FLAGS.label_translation_file)
    else:
        label_map = None

    # Process files
    commands = []

    for input_f, output_f in zip(filenames, output_filenames):
        commands.append((
            input_f,
            output_f,
            locations,
            label_map,
            class_labels,
            translate_labels,
        ))

    if FLAGS.jobs == 1:
        for command in commands:
            process_file(*command, verbose=True)
    else:
        jobs = FLAGS.jobs if FLAGS.jobs != 0 else None
        run_job_pool(process_file, commands, cores=jobs)


if __name__ == "__main__":
    app.run(main)
