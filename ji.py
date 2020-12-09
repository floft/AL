#!/usr/bin/python

# python oc.py <data_file>+
#
# Performs activity learning on the given data files and outputs either the
# learned model, or the confusion matrices and accuracy for a 3-fold
# cross-validation test. Each activity is learned by a separate
# one-class classifier.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os.path
import sys
from copy import deepcopy

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

import activity
import config
import features
import loc
import person
import utils


class WD:

    def __init__(self, conf: config.Config):
        """ Constructor
        """
        self.conf = conf
        self.aclass = activity.Activity()
        self.location = loc.Location(conf=self.conf)
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.rotx = []
        self.roty = []
        self.rotz = []
        self.accx = []
        self.accy = []
        self.accz = []
        self.acctotal = []
        self.latitude = []
        self.longitude = []
        self.altitude = []
        self.course = []
        self.speed = []
        self.hacc = []
        self.vacc = []
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0
        self.write = True
        self.oc = True
        return

    def resetvars(self):
        """ Initialize the feature arrays.
        """
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.rotx = []
        self.roty = []
        self.rotz = []
        self.accx = []
        self.accy = []
        self.accz = []
        self.acctotal = []
        self.latitude = []
        self.longitude = []
        self.altitude = []
        self.course = []
        self.speed = []
        self.hacc = []
        self.vacc = []
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0
        return 0

    def update_location_range(self, value, datatype):
        """ Maintain min and max latitude and longitude values to compute relative
        distances for each person.
        """
        if datatype == "latitude":
            minrange = self.minlat
            maxrange = self.minlat
        else:
            minrange = self.minlong
            maxrange = self.minlong
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
            self.minlat = maxrange
        else:
            self.minlong = minrange
            self.minlong = maxrange
        return

    def read_sensors_from_window(self, lines):
        """ Read and store one set of sensor readings.
        """
        self.resetvars()
        for i in range(self.conf.numseconds):
            line = lines[i * self.conf.numsensors + 0]
            self.yaw.append(utils.clean(float(line[4]), -5.0, 5.0))
            line = lines[i * self.conf.numsensors + 1]
            self.pitch.append(utils.clean(float(line[4]), -5.0, 5.0))
            line = lines[i * self.conf.numsensors + 2]
            self.roll.append(utils.clean(float(line[4]), -5.0, 5.0))
            line = lines[i * self.conf.numsensors + 3]
            self.rotx.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 4]
            self.roty.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 5]
            self.rotz.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 6]
            v1 = utils.clean(float(line[4]), -1.0, 1.0)
            self.accx.append(v1)
            temp = v1 * v1
            line = lines[i * self.conf.numsensors + 7]
            v1 = utils.clean(float(line[4]), -1.0, 1.0)
            self.accy.append(v1)
            temp += v1 * v1
            line = lines[i * self.conf.numsensors + 8]
            v1 = utils.clean(float(line[4]), -1.0, 1.0)
            self.accz.append(v1)
            temp += v1 * v1
            self.acctotal.append(np.sqrt(temp))
            line = lines[i * self.conf.numsensors + 9]
            self.latitude.append(float(line[4]))
            self.update_location_range(float(line[4]), datatype="latitude")
            line = lines[i * self.conf.numsensors + 10]
            self.longitude.append(float(line[4]))
            self.update_location_range(float(line[4]), datatype="longitude")
            line = lines[i * self.conf.numsensors + 11]
            self.altitude.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 12]
            self.course.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 13]
            self.speed.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 14]
            self.hacc.append(float(line[4]))
            line = lines[i * self.conf.numsensors + 15]
            self.vacc.append(float(line[4]))
            dt = utils.get_datetime(line[0], line[1])
            if i == (self.conf.numseconds - 1):
                label = line[5]
        return dt, label


def writedata(datafilename, wd, cf):
    xdata = []
    lines = []
    infile = os.path.join(cf.datapath, datafilename + cf.extension)
    outfile = os.path.join(cf.datapath, datafilename + '.csv')

    personfile = os.path.join(cf.datapath, datafilename + '.person')
    if not os.path.isfile(personfile):
        print(personfile, "does not exist, generating these stats")
        person.main(datafilename, cf)
    person_stats = np.loadtxt(personfile, delimiter=',')
    oc_clusters = features.load_clusters(datafilename, cf)

    count = 0
    for line in open(infile):
        count += 1
        lines.append(utils.process_entry(line))
        # collect one set of sensor readings
        if (count % cf.numsensors) == 0 and count >= cf.windowsize:
            dt, alabel = wd.read_sensors_from_window(lines)
            map_alabel1 = wd.aclass.map_activity_name(alabel)
            map_alabel2 = wd.aclass.map_activity_name2(alabel)
            xpoint = features.create_point(wd, dt, infile, person_stats, oc_clusters)
            if xpoint[-4] == 1:  # add location type
                locpoint = 1
            elif xpoint[-3] == 1:
                locpoint = 2
            elif xpoint[-2] == 1:
                locpoint = 3
            else:
                locpoint = 4
            xpoint[-4] = locpoint
            xpoint = xpoint[:-3]
            newlabels = []
            if wd.oc:  # add oneclass values
                for i in range(cf.numactivities - 1):
                    if alabel == cf.activities[i] or \
                            map_alabel1 == cf.activities[i] or \
                            map_alabel2 == cf.activities[i]:
                        newlabels.append(1)
                    else:
                        newlabels.append(0)
            if map_alabel2 != 'Ignore' and map_alabel2 is not None and \
                    map_alabel2 != 'Other':  # add activity value
                newlabels.append(cf.activity_list.index(map_alabel2))
                xpoint += newlabels
                xdata.append(xpoint)
            lines = lines[cf.numsensors:]
    outdata = np.array(xdata)
    np.savetxt(outfile, outdata, delimiter=',')


def joint_predict(base_filename, cf):
    """ Target variables are location (index 628), oc activities (629-661), and
    activity(662).
    """
    clf = RandomForestClassifier(n_estimators=100, bootstrap=True,
                                 criterion="entropy", class_weight="balanced",
                                 max_depth=10, n_jobs=20)
    # low = 628
    low = 629
    infile = os.path.join(cf.datapath, base_filename + '.csv')
    data = np.loadtxt(fname=infile, delimiter=',')
    n = len(data)
    k = 663 - low + 1
    kf = KFold(n_splits=3)
    results = np.zeros((k, 3))
    for i in range(low, 663):  # none or all location, oc or activity ground truth
        xdata = deepcopy(data)
        xdata = np.delete(xdata, i, axis=1)
        ydata = data[:, i]
        score = 0
        for train, test in kf.split(data):
            xtraindata = xdata[train]
            xtestdata = xdata[test]
            ytraindata = ydata[train]
            ytestdata = ydata[test]
            clf.fit(xtraindata, ytraindata)
            newlabels = clf.predict(xtestdata)
            score += metrics.accuracy_score(ytestdata, newlabels)
        results[i - low][1] = score / 3.0
        shortxdata = xdata[:, :low]
        score = 0
        for train, test in kf.split(data):
            xtraindata = shortxdata[train]
            xtestdata = shortxdata[test]
            ytraindata = ydata[train]
            ytestdata = ydata[test]
            clf.fit(xtraindata, ytraindata)
            newlabels = clf.predict(xtestdata)
            score += metrics.accuracy_score(ytestdata, newlabels)
        results[i - low][0] = score / 3.0
        newdata = deepcopy(data)
        clf.fit(shortxdata, ydata)
        newlabels = clf.predict(shortxdata)
        newdata[:, i] = newlabels  # replace ground truth with inferred information
    for i in range(low, 663):  # only inferred location, or, activity information
        xdata = deepcopy(newdata)
        xdata = np.delete(xdata, i, axis=1)
        ydata = data[:, i]
        score = 0
        for train, test in kf.split(newdata):
            xtraindata = newdata[train]
            xtestdata = newdata[test]
            ytraindata = ydata[train]
            ytestdata = ydata[test]
            clf.fit(xtraindata, ytraindata)
            newlabels = clf.predict(xtestdata)
            score += metrics.accuracy_score(ytestdata, newlabels)
        results[i - low][2] = score / 3.0
        print('target [none perfect joint]', i, 'score', results[i - low])


def main():
    """ Run with write=True to generate csv files and with write=False to
    create scores.
    """
    cf = config.Config(description='WD in ji.py')
    cf.locmodel = 0
    cf.set_parameters()

    wd = WD(conf=cf)
    wd.aclass.read_activity_mappings_both()
    wd.location.read_location_mappings()
    locations_index = wd.location.read_locations()

    for base_filename in cf.files:
        if wd.write:
            writedata(base_filename, wd, cf)
        else:
            joint_predict(base_filename, cf)


if __name__ == "__main__":
    main()
