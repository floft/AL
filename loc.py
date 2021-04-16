#!/usr/bin/python

# python Location.py <data_file>
#
# Performs location type learning on the given data file and outputs either the
# learned model, or the confusion matrices and accuracy for a 3-fold
# cross-validation test.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import math
import os.path
from datetime import datetime, timedelta
from statistics import mean
from typing import Optional, Dict, Union, List

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config
import features
import gps
import utils
from base import BaseDataProcessor
from mobiledata import MobileData


class Location(BaseDataProcessor):
    """
    Class for processing data for a Location model and finding location types, etc.
    """

    def __init__(self, conf: config.Config):
        """
        Override the default initializer to set up other location specific items, such as location
        mappings, the locations filename, and the classifier.

        Parameters
        ----------
        conf : config.Config
            An AL Config object that has been loaded with the configuration to use
            Gets attached to this object
        """

        super().__init__(conf)

        # Set up default location mappings:
        self.lmappings = dict()
        self.lmappings['other'] = 'other'

        # Set up the list of known locations:
        self.locations = list()

        # Create the location classifier:
        self.clf = RandomForestClassifier(n_estimators=50,
                                          bootstrap=True,
                                          criterion="entropy",
                                          class_weight="balanced",
                                          max_depth=5,
                                          n_jobs=self.conf.loc_n_jobs)

    def read_location_mappings(self):
        """
        Generate a translate list for location names.
        This function assumes that file loc.translate exists in the same directory as the code.
        File loc.translate contains an arbitrary number of lines, each with syntax
        "specificType mappedType". This function maps locations of specificType to the
        corresponding, more general, mappedType.
        """

        with open('loc.translate', 'r') as file:
            for line in file:
                x = str(str(line).strip()).split(' ', 2)
                self.lmappings[x[0]] = x[1]

    def map_location_name(self, name: str) -> str:
        """
        Return the location type that is associated with a specific location name, using the stored
        list of location mappings.

        Parameters
        ----------
        name : str
            The location type name to map

        Returns
        -------
        str
            The mapped location name (or 'other' if not found in mappings)
        """

        newname = self.lmappings.get(name)
        if newname is None:
            return 'other'
        else:
            return newname

    def read_locations(self):
        """
        Read and store list of locations and corresponding location types.

        This function assumes that file locations exists in the same directory as the code. File
        locations contains an arbitrary number of lines, each with syntax
        "latitude longitude type1 type2 type3". Open street maps return as many as three location
        types associated with a lat,long location. They can provide alternate type names or levels
        of abstraction.

        In this function, only the first type is stored with the latitude and longitude.
        """

        if os.path.isfile('locations'):
            with open('locations', 'r') as file:
                for line in file:
                    x = str(line.strip()).split(' ', 4)

                    loc_tuple = list()

                    loc_tuple.append(float(x[0]))
                    loc_tuple.append(float(x[1]))

                    loc_tuple.append(x[2])
                    loc_tuple.append(x[3])
                    loc_tuple.append(x[4])

                    self.locations.append(loc_tuple)

    def find_location(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Determine whether the input location is close (within a threshold distance) to the
        locations already stored in the external list. Return closest location within the threshold,
        or None if none exist.

        Parameters
        ----------
        latitude : float
            Latitude value of location to find
        longitude : float
            Longitude value of location to find

        Returns
        -------
        Optional[str]
            The location type if found, else None
        """

        threshold = 0.005
        find_loc = None

        for loc_tuple in self.locations:
            tlat = loc_tuple[0]
            tlong = loc_tuple[1]

            dist = math.sqrt(((tlat - latitude) * (tlat - latitude)) +
                             ((tlong - longitude) * (tlong - longitude)))

            if dist < threshold:
                threshold = dist
                find_loc = loc_tuple[2]

        return find_loc

    def get_location_type(self, latitude: float, longitude: float) -> str:
        """
        Determine the location type for a given latitude/longitude location.

        If the location is in the existing list of locations, return its type. Otherwise, try to
        reverse geocode the location, then add it to the list of locations and return the found
        type.

        Parameters
        ----------
        latitude : float
            Latitude of the location to get type of
        longitude : float
            Longitude of the location to get type of

        Returns
        -------
        str
            The type of the location (not translated)
        """

        existing_loc_type = self.find_location(latitude, longitude)

        if existing_loc_type is not None:
            return existing_loc_type
        else:
            # Couldn't find existing location, so reverse-geocode it:
            gps_loc = list()

            gps_loc.append(latitude)
            gps_loc.append(longitude)

            # Get the reverse-geocoded location type
            gps_type = gps.get_location_type(gps_loc, 'locations')

            gps_loc.append(gps_type)
            gps_loc.append(gps_type)
            gps_loc.append(gps_type)

            # Add the location tuple to the list:
            self.locations.append(gps_loc)

            return gps_type

    def new_window(self, delta: timedelta) -> bool:
        """
        Override the new_window method to ignore annotate value that base looks at.
        i.e. start new window if there is a 2-second gap, we just generated a window, or
        we've reached a full `samplesize` events in the window.
        """

        return delta.seconds > 2 or self.generated_window or self.count % self.conf.samplesize == 0

    def should_create_feats_for_window(self,
                                       latest_event: Dict[str, Union[datetime, float, str, None]]) \
            -> bool:
        """
        Override to check that we have valid location data in the window.
        """

        return self.valid_location_data(self.latitude, self.longitude, self.altitude)

    def process_window_feats(self,
                             latest_event: Dict[str, Union[datetime, float, str, None]],
                             feats: List[float]
                             ):
        """
        Called when we have a feature vector at end of window.

        Try to find the GPS location type. If it's not `'None'`, then append the feature vector
        to `self.xdata` and add the location type mapping to `self.ydata`.

        Parameters
        ----------
        latest_event : Dict[str, Union[datetime, float, str, None]]
            The latest event in the window. Not used here
        feats : List[float]
            The feature vector created for the window
        """

        # Find the location type for the mean lat/lon values from the window:
        loc_type = self.get_location_type(mean(self.latitude), mean(self.longitude))

        # If the location type is valid, append the feature vector and the mapping of the type to
        # our training data:
        if loc_type != 'None':
            self.xdata.append(feats)

            y_value = self.map_location_name(loc_type)
            self.ydata.append(y_value)

    @staticmethod
    def valid_location_data(latitude: List[float], longitude: List[float], altitude: List[float]) \
            -> bool:
        """
        Check if location data (namely latitude) is valid in the provided lists of sensor values.

        If there are valid latitude values, then replace any invalid location points with the first
        valid one.

        Parameters
        ----------
        latitude : List[float]
            Latitude values to check
        longitude : List[float]
            Longitude values to check
        altitude : List[float]
            Altitude values to check

        Returns
        -------
        bool
            True if at least one valid latitude was found
        """

        n = len(latitude)

        valid_lat = valid_long = valid_alt = 0.0

        for i in range(n):
            if -90.0 < latitude[i] < 90.0:
                valid_lat = latitude[i]
                valid_long = longitude[i]
                valid_alt = altitude[i]

        if valid_lat == 0.0:
            return False
        else:
            for i in range(n):
                if latitude[i] <= -90.0 or latitude[i] >= 90.0:
                    latitude[i] = valid_lat
                    longitude[i] = valid_long
                    altitude[i] = valid_alt

        return True


class OLDLocation:

    def __init__(self, conf: config.Config, filename: str = None):
        """ Constructor
        """
        self.lmappings = dict()
        self.lmappings['other'] = 'other'
        if filename is None:
            self.infile = "locations"
        else:
            self.infile = filename
        self.conf = conf
        self.yaw = list()
        self.pitch = list()
        self.roll = list()
        self.rotx = list()
        self.roty = list()
        self.rotz = list()
        self.accx = list()
        self.accy = list()
        self.accz = list()
        self.acctotal = list()
        self.latitude = list()
        self.longitude = list()
        self.altitude = list()
        self.course = list()
        self.speed = list()
        self.hacc = list()
        self.vacc = list()
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0
        self.locations = list()
        self.xdata = list()
        self.ydata = list()
        self.clf = RandomForestClassifier(n_estimators=50,
                                          bootstrap=True,
                                          criterion="entropy",
                                          class_weight="balanced",
                                          max_depth=5,
                                          n_jobs=self.conf.loc_n_jobs)
        return

    def map_location_name(self, name):
        """ Return the location type that is associated with a specific location
        name, using the stored list of location mappings.
        """
        newname = self.lmappings.get(name)
        if newname is None:
            return 'other'
        else:
            return newname

    @staticmethod
    def generate_location_features(name):
        """ Transform a location type into a vector using one-shot encoding.
        The location types are house, road, work, or other.
        """
        if name == 'house':
            return 1, 0, 0, 0
        elif name == 'road':
            return 0, 1, 0, 0
        elif name == 'work':
            return 0, 0, 1, 0
        else:
            return 0, 0, 0, 1

    def read_location_mappings(self):
        """ Generate a translate list for location names.
        This function assumes that file loc.translate exists in the same
        directory as the code. File loc.translate contains an arbitrary number
        of lines, each with syntax "specificType mappedType". This function maps
        locations of specificType to the corresponding, more general, mappedType.
        """
        with open('loc.translate', "r") as file:
            for line in file:
                x = str(str(line).strip()).split(' ', 2)
                self.lmappings[x[0]] = x[1]

    def read_locations(self):
        """ Read and store list of locations and corresponding location types.
        This function assumes that file locations exists in the same directory
        as the code. File locations contains an arbitrary number of lines, each
        with syntax "latitude longitude type1 type2 type3". Open street maps
        return as many as three location types associated with a lat,long
        location. They can provide alternate type names or levels of abstraction.
        In this function, only the first type is stored with the latitude
        and longitude.
        """
        new_locations_index = 0
        if os.path.isfile('locations'):
            with open('locations', "r") as file:
                for line in file:
                    x = str(str(line).strip()).split(' ', 4)
                    loc_tuple = list()
                    loc_tuple.append(float(x[0]))
                    loc_tuple.append(float(x[1]))
                    loc_tuple.append(x[2])
                    loc_tuple.append(x[3])
                    loc_tuple.append(x[4])
                    self.locations.append(loc_tuple)
                    new_locations_index = new_locations_index + 1
        return new_locations_index

    def find_location(self, latitude, longitude):
        """ Determine whether the input location is close (within a threshold
        distance) to the locations already stored in the external list. Return
        closest location within the thresold, or None if none exist.
        """
        threshold = 0.005
        find_loc = None
        for loc_tuple in self.locations:
            tlat = loc_tuple[0]
            tlong = loc_tuple[1]
            dist = math.sqrt(((tlat - latitude) * (tlat - latitude)) +
                             ((tlong - longitude) * (tlong - longitude)))
            if dist < threshold:
                threshold = dist
                find_loc = loc_tuple[2]
        return find_loc

    def generate_gps_features(self, latitude, longitude):
        """ Generate location features.
        """
        gps_loc = self.find_location(latitude, longitude)
        if gps_loc is not None:
            return gps_loc
        else:
            gps_loc = list()
            gps_loc.append(latitude)
            gps_loc.append(longitude)
            gps_type = gps.get_location_type(gps_loc, 'locations')
            gps_loc.append(gps_type)
            gps_loc.append(gps_type)
            gps_loc.append(gps_type)
            self.locations.append(gps_loc)
            return gps_type

    def resetvars(self):
        """ Initialize the feature arrays.
        """
        self.yaw = list()
        self.pitch = list()
        self.roll = list()
        self.rotx = list()
        self.roty = list()
        self.rotz = list()
        self.accx = list()
        self.accy = list()
        self.accz = list()
        self.acctotal = list()
        self.latitude = list()
        self.longitude = list()
        self.altitude = list()
        self.course = list()
        self.speed = list()
        self.hacc = list()
        self.vacc = list()
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0
        return 0

    def new_window(self, delta, gen, count):
        """ Determine if conditions are met to start a new window.
        """
        return ((delta.seconds > 2) or (gen == 1) or
                (count % self.conf.samplesize) == 0)

    def update_location_range(self, value, datatype):
        """ Maintain min and max latitude and longitude values to compute relative
        distances for each person.
        """
        if datatype == "latitude":
            minrange = self.minlat
            maxrange = self.maxlat
        else:
            minrange = self.minlong
            maxrange = self.maxlong

        if not minrange:
            minrange = float(value)
        elif float(value) < minrange:
            minrange = float(value)

        if not maxrange:
            maxrange = float(value)
        elif float(value) > maxrange:
            maxrange = float(value)

        if datatype == "latitude":
            self.minlat = minrange
            self.maxlat = maxrange
        else:
            self.minlong = minrange
            self.maxlong = maxrange

        return

    def update_sensors(self, event: Dict[str, Union[datetime, float, str, None]]):
        """
        Update the sensor lists based on the input event.

        Parameters
        ----------
        event : Dict[str, Union[datetime, float, str, None]]
            An event dictionary with fields mapped to their respective values
            Normally these would be output from the CSV Data Layer
        """

        # Set all None values for sensors to just be zeros
        # This just creates a new dictionary setting the value for each field to 0.0 if it is None:
        e = {f: v if v is not None else 0.0 for f, v in event.items()}

        yaw_clean = utils.clean(e['yaw'], -5.0, 5.0)
        self.yaw.append(yaw_clean)

        pitch_clean = utils.clean(e['pitch'], -5.0, 5.0)
        self.pitch.append(pitch_clean)

        roll_clean = utils.clean(e['roll'], -5.0, 5.0)
        self.roll.append(roll_clean)

        self.rotx.append(e['rotation_rate_x'])
        self.roty.append(e['rotation_rate_y'])
        self.rotz.append(e['rotation_rate_z'])

        acc_x_clean = utils.clean(e['user_acceleration_x'], -1.0, 1.0)
        self.accx.append(acc_x_clean)
        acc_total = acc_x_clean * acc_x_clean

        acc_y_clean = utils.clean(e['user_acceleration_y'], -1.0, 1.0)
        self.accy.append(acc_y_clean)
        acc_total += acc_y_clean * acc_y_clean

        acc_z_clean = utils.clean(e['user_acceleration_z'], -1.0, 1.0)
        self.accz.append(acc_z_clean)
        acc_total += acc_z_clean * acc_z_clean

        self.acctotal.append(math.sqrt(acc_total))  # compute combined acceleration

        self.latitude.append(e['latitude'])
        self.update_location_range(e['latitude'], datatype="latitude")

        self.longitude.append(e['longitude'])
        self.update_location_range(e['longitude'], datatype="longitude")

        self.altitude.append(e['altitude'])

        self.course.append(e['course'])
        self.speed.append(e['speed'])
        self.hacc.append(e['horizontal_accuracy'])
        self.vacc.append(e['vertical_accuracy'])

    def extract_features(self, base_filename):
        """ Extract a feature vector that will be input to a location classifier.
        """

        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        in_data = MobileData(infile, 'r')
        in_data.open()

        # Shorthand access to stamp field name:
        stamp_field = self.conf.stamp_field_name

        count = 0

        prevdt = None  # type: Optional[datetime]

        gen = self.resetvars()

        # Loop over all event rows in the input file:
        for event in in_data.rows_dict:
            # Get event's stamp and use that to compute delta since last event:
            dt = event[stamp_field]

            # Set prevdt to this time if None (first event):
            if prevdt is None:
                prevdt = dt

            delta = dt - prevdt

            if self.new_window(delta, gen, count):  # start new window
                gen = self.resetvars()

            # Update the sensor values for this window:
            self.update_sensors(event)

            if (count % self.conf.samplesize) == (self.conf.samplesize - 1):  # end of window
                xpoint = list()
                gen = 1

                if self.valid_location_data(self.latitude, self.longitude, self.altitude):
                    for i in [self.yaw, self.pitch, self.roll, self.rotx, self.roty,
                              self.rotz, self.accx, self.accy, self.accz, self.acctotal]:
                        xpoint.extend(features.generate_features(i, self.conf))

                    for i in [self.latitude, self.longitude, self.altitude]:
                        # Only include absolute features if enabled in config:
                        xpoint.extend(features.generate_features(i, self.conf, include_absolute_features=self.conf.gen_gps_abs_stat_features))

                    for i in [self.course, self.speed, self.hacc, self.vacc]:
                        xpoint.extend(features.generate_features(i, self.conf))

                    month, dayofweek, hours, minutes, seconds, distance, hcr, sr, trajectory = \
                        features.calculate_time_and_space_features(self, dt)

                    xpoint.append(distance)
                    xpoint.append(hcr)
                    xpoint.append(sr)
                    xpoint.append(trajectory)

                    xpoint.append(month)
                    xpoint.append(dayofweek)
                    xpoint.append(hours)
                    xpoint.append(minutes)
                    xpoint.append(seconds)

                    place = self.generate_gps_features(mean(self.latitude),
                                                       mean(self.longitude))

                    if place != 'None':
                        self.xdata.append(xpoint)

                        yvalue = self.map_location_name(place)
                        self.ydata.append(yvalue)

            prevdt = dt

            count += 1

            if (count % 100000) == 0:
                print('count', count)

        in_data.close()

        return

    def label_loc(self, st, distance, hcr, sr, trajectory,
                  month, dayofweek, hours, minutes, seconds):
        """ Use the pretrained location classifier to extract features from the
        input sensor values and map the feature vector onto a location type.
        """

        xpoint = list()

        for i in [st.yaw, st.pitch, st.roll, st.rotx, st.roty, st.rotz, st.accx,
                  st.accy, st.accz, st.acctotal]:
            xpoint.extend(features.generate_features(i, st.conf))

        for i in [st.latitude, st.longitude, st.altitude]:
            # Only include absolute features if enabled in config:
            xpoint.extend(features.generate_features(i, st.conf, include_absolute_features=st.conf.gen_gps_abs_stat_features))

        for i in [st.course, st.speed, st.hacc, st.vacc]:
            xpoint.extend(features.generate_features(i, st.conf))

        xpoint.append(distance)
        xpoint.append(hcr)
        xpoint.append(sr)
        xpoint.append(trajectory)

        xpoint.append(month)
        xpoint.append(dayofweek)
        xpoint.append(hours)
        xpoint.append(minutes)
        xpoint.append(seconds)

        self.xdata = [xpoint]

        if st.locclf is None:
            return 'other'
        else:
            labels = st.locclf.predict(self.xdata)
            return labels[0]

    def train_location_model(self):
        """ Train a model to map a feature vector (statistical operations
        applied to sensor values and raw location values) onto a location type.
        """
        aset = set(self.ydata)
        # Store the learned model.
        self.clf.fit(self.xdata, self.ydata)
        filename = os.path.join(self.conf.modelpath, 'locmodel.pkl')
        joblib.dump(self.clf, filename)
        return

    def cross_validation(self):
        for i in range(self.conf.cv):
            numright = 0
            total = 0
            xtrain, xtest, ytrain, ytest = train_test_split(self.xdata,
                                                            self.ydata,
                                                            test_size=0.33,
                                                            random_state=i)
            self.clf.fit(xtrain, ytrain)
            newlabels = self.clf.predict(xtest)
            print('newlabels', newlabels)
            matrix = confusion_matrix(ytest, newlabels)
            print('matrix', matrix)
            for j in range(len(ytest)):
                if newlabels[j] == ytest[j]:
                    numright += 1
                total += 1
            print('accuracy', float(numright) / float(total))
        return

    def load_location_model(self):
        """ Load a pretrained model that maps a feature vector
        (statistical operations applied to sensor values and raw location values)
        onto a location type.
        """
        filename = os.path.join(self.conf.modelpath, 'locmodel.pkl')
        if not os.path.isfile(filename):
            print("no location model", filename, "will use location other")
            return None
        else:
            clf = joblib.load(filename)
            return clf

    @staticmethod
    def valid_location_data(latitude, longitude, altitude):
        """ Do not include the training data point if location values are
        out of range.
        """
        n = len(latitude)
        valid_lat = valid_long = valid_alt = 0.0
        for i in range(n):
            if -90.0 < latitude[i] < 90.0:
                valid_lat = latitude[i]
                valid_long = longitude[i]
                valid_alt = altitude[i]
        if valid_lat == 0.0:
            return False
        else:
            for i in range(n):
                if latitude[i] <= -90.0 or latitude[i] >= 90.0:
                    latitude[i] = valid_lat
                    longitude[i] = valid_long
                    altitude[i] = valid_alt
        return True


if __name__ == "__main__":
    """ Syntax is python loc.py trainingdata. The set of latitude longitude
    locations with corresponding ground truth location type is assumed to be
    in file locations.
    """
    cf = config.Config(description='Location Model Training')
    # Set the default mode to train.
    cf.mode = config.MODE_TRAIN_MODEL
    cf.set_parameters()
    loc = Location(conf=cf)
    if cf.translate:
        loc.read_location_mappings()
    locations_index = loc.read_locations()
    if cf.mode in [config.MODE_TRAIN_MODEL, config.MODE_CROSS_VALIDATION]:
        for datafile in cf.files:
            loc.extract_features(datafile)
        print('update', len(loc.locations))
        gps.update_locations(loc.locations, 'locations')
        if cf.mode == config.MODE_TRAIN_MODEL:
            print('train')
            loc.train_location_model()
        elif cf.mode == config.MODE_CROSS_VALIDATION:
            print('cross_validation')
            loc.cross_validation()
    else:
        print('This mode is not supported for Locations.')

