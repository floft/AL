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
import sys

import joblib
import numpy as np
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config
import features
import gps
import utils


class Location:

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

    @staticmethod
    def read_entry(infile):
        """ Parse a single line from a text file containing a sensor reading.
        The format is "date time sensorname sensorname value <activitylabel|0>".
        """
        try:
            line = infile.readline()
            x = str(str(line).strip()).split(' ', 5)
            if len(x) < 6:
                return True, x[0], x[1], x[2], x[3], x[4], 'None'
            else:
                x[5] = x[5].replace(' ', '_')
                return True, x[0], x[1], x[2], x[3], x[4], x[5]
        except:
            return False, None, None, None, None, None, None

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
    def generate_location_num(name):
        """ Transform a location type into an index value.
        """
        if name == 'house':
            return 0
        elif name == 'road':
            return 1
        elif name == 'work':
            return 2
        else:
            return 3

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

    def read_sensors(self, infile, v1):
        """ Read and store one set of sensor readings.
        The first line is already read.
        """
        self.yaw.append(utils.clean_range(float(v1), -5.0, 5.0))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.pitch.append(utils.clean_range(float(v1), -5.0, 5.0))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.roll.append(utils.clean_range(float(v1), -5.0, 5.0))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.rotx.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.roty.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.rotz.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accx.append(v1)
        temp = v1 * v1
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accy.append(v1)
        temp += v1 * v1
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accz.append(v1)
        temp += v1 * v1
        self.acctotal.append(np.sqrt(temp))  # compute combined acceleration

        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.latitude.append(float(v1))
        self.update_location_range(float(v1), datatype="latitude")
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.longitude.append(float(v1))
        self.update_location_range(float(v1), datatype="longitude")
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.altitude.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.course.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.speed.append(float(v1))
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.hacc.append(float(v1))
        pdt = utils.get_datetime(date, sen_time)
        valid, date, sen_time, f1, f2, v1, v2 = self.read_entry(infile)
        self.vacc.append(float(v1))
        return pdt, v2, date, sen_time

    def extract_features(self, base_filename):
        """ Extract a feature vector that will be input to a location classifier.
        """
        fs1 = list()
        fs2 = list()

        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        features_datafile = open(infile, "r")  # process input file to create feature vector

        count = 0

        valid, date, feat_time, f1, f2, v1, v2 = self.read_entry(features_datafile)

        prevdt = utils.get_datetime(date, feat_time)

        gen = self.resetvars()

        while valid:
            dt = utils.get_datetime(date, feat_time)
            delta = dt - prevdt

            if self.new_window(delta, gen, count):  # start new window
                gen = self.resetvars()

            pdt, v2, date, feat_time = self.read_sensors(features_datafile, v1)

            month, dayofweek, hours, minutes, seconds, distance, hcr, sr, trajectory = \
                features.calculate_time_and_space_features(self, dt)

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

            if not valid:
                prevdt = pdt
            else:
                prevdt = utils.get_datetime(date, feat_time)

            count += 1

            if (count % 100000) == 0:
                print('count', count)

            valid, date, feat_time, f1, f2, v1, v2 = self.read_entry(features_datafile)

        features_datafile.close()
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

