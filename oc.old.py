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


import datetime
import os.path
import sys
from datetime import datetime
from multiprocessing import Process, Queue

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import activity
import config
import features
import gps
import loc
import person
import utils


class OC:

    def __init__(self, conf: config.Config):
        """ Constructor
        """
        self.conf = conf
        self.aclass = activity.Activity()
        self.location = loc.Location(conf=self.conf)
        self.clf = RandomForestClassifier(n_estimators=100,
                                          bootstrap=True,
                                          criterion="entropy",
                                          class_weight="balanced",
                                          max_depth=10,
                                          n_jobs=self.conf.oc_n_jobs)
        self.locclf = None
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
        self.alt = list()
        self.course = list()
        self.speed = list()
        self.hacc = list()
        self.vacc = list()
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0
        return

    def cross_validate_model(self, xdata: list, ydata: list):
        """ Perform k-fold cross validation on the modesl.
        """
        aset = set(ydata)  # list of activity names
        for model_activity in aset:
            binary_values = np.full(len(ydata), 0)
            for i in range(len(ydata)):  # Binary values based on target activity
                if ydata[i] == model_activity:
                    binary_values[i] = 1

            taccuracy = tprecision = trecall = tf1score = 0
            for i in range(self.conf.cv):
                tp = tn = fp = fn = total = 0
                xtrain, xtest, ytrain, ytest = train_test_split(xdata,
                                                                binary_values,
                                                                test_size=0.33,
                                                                random_state=i)
                self.clf.fit(xtrain, ytrain)
                newlabels = self.clf.predict(xtest)
                if i == (self.conf.cv - 1):
                    matrix = confusion_matrix(ytest, newlabels)
                    print('fold', i + 1, '\n', matrix)
                for j in range(len(ytest)):
                    if newlabels[j] == ytest[j]:
                        if newlabels[j] == 1:
                            tp += 1
                        else:
                            tn += 1
                    elif newlabels[j] == 1:
                        fp += 1
                    else:
                        fn += 1
                    total += 1
                accuracy = float(tp + tn) / float(total)
                if (tp + fp) == 0:
                    precision = 1.0
                else:
                    precision = float(tp) / float(tp + fp)
                if (tp + fn) == 0:
                    recall = 1.0
                else:
                    recall = float(tp) / float(tp + fn)
                if (precision + recall) == 0:
                    f1score = 0.0
                else:
                    f1score = (2.0 * precision * recall) / (precision + recall)
                taccuracy += accuracy
                tprecision += precision
                trecall += recall
                tf1score += f1score
            taccuracy /= float(self.conf.cv)
            tprecision /= float(self.conf.cv)
            trecall /= float(self.conf.cv)
            tf1score /= float(self.conf.cv)
            print("A: %5.3f, P: %5.3f, R: %5.3f, F1: %5.3f" %
                  (taccuracy, tprecision, trecall, tf1score))
        return

    def train_model(self, xdata: list, ydata: list):
        """ Train a set of one-class activity models using
            features and labels extracted from training data.
        """
        aset = set(ydata)  # list of activity names
        for model_activity in aset:
            binary_values = np.full(len(ydata), 0)
            for i in range(len(ydata)):  # Binary values based on target activity
                if ydata[i] == model_activity:
                    binary_values[i] = 1

            self.clf.fit(xdata, binary_values)
            outstr = self.conf.modelpath + model_activity + ".pkl"
            joblib.dump(self.clf, outstr)
        return

    def test_model(self, xdata: list, ydata: list):
        print('This feature is not yet implemented.')
        exit()
        return

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

    def read_sensors_from_window(self, lines):
        """ Read and store one set of sensor readings.
        """
        self.resetvars()
        dt = None
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
        return dt

    def read_sensors(self, infile, v1):
        """ Read and store one set of sensor readings.
        """
        self.yaw.append(utils.clean(float(v1), -5.0, 5.0))  # first line already read
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.pitch.append(utils.clean(float(v1), -5.0, 5.0))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.roll.append(utils.clean(float(v1), -5.0, 5.0))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.rotx.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.roty.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.rotz.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accx.append(v1)
        temp = v1 * v1
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accy.append(v1)
        temp += v1 * v1
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accz.append(v1)
        temp += v1 * v1
        self.acctotal.append(np.sqrt(temp))  # compute combined acceleration

        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.latitude.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.longitude.append(float(v1))
        self.update_location_range(float(v1), datatype="latitude")
        self.update_location_range(float(v1), datatype="longitude")
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.altitude.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.course.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.speed.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.hacc.append(float(v1))
        pdt = utils.get_datetime(date, time)
        valid, date, time, f1, f2, v1, v2 = self.location.read_entry(infile)
        self.vacc.append(float(v1))
        label = self.aclass.map_activity_name(v2)
        label2 = self.aclass.map_activity_name2(v2)
        return pdt, v2, date, time, label, label2

    def output_combined_window(self, outfile, count, lines, newlabels):
        """ Write a single window of annotated sensor readings as a single csv line
        to a file.
        """
        dt1 = datetime.strptime("2010-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        if count == self.conf.windowsize:  # Process beginning before full window
            start = 0
        else:
            start = self.conf.samplesize - 1
        for i in range(start, self.conf.samplesize):
            for j in range(self.conf.numsensors):
                offset = i * self.conf.numsensors
                outstr = ''
                if j == 0:
                    dt2 = datetime.strptime(lines[offset + j][0] + " 00:00:00",
                                            "%Y-%m-%d %H:%M:%S")
                    dt = utils.get_datetime(lines[offset + j][0], lines[offset + j][1])
                    delta = dt - dt1
                    days = delta.days
                    delta = dt - dt2
                    seconds = delta.seconds
                    outstr = str(days) + ',' + str(seconds) + ','
                outstr += lines[offset + j][4]
                if j < (self.conf.numsensors - 1):
                    outstr += ','
                else:
                    for k in range(self.conf.numactivities):
                        outstr += ',' + str(newlabels[k])
                    outstr += '\n'
                    outfile.write(outstr)
        return

    def load_models(self):
        """ Load the pre-trained person-specific one-class activity models.
        """
        oc_clfs = list()
        for i in range(self.conf.numactivities):
            if i == (self.conf.numactivities - 1):
                # Most likely activity.
                instr = os.path.join(self.conf.modelpath, 'model.pkl')
                if not os.path.isfile(instr):
                    print(instr, "model file does not exist")
                    exit()
                oc_clfs.append(joblib.load(instr))
            else:  # one-class activity
                instr = os.path.join(self.conf.modelpath, self.conf.activities[i] + '.pkl')
                if not os.path.isfile(instr):
                    print(instr, "model file does not exist")
                    exit()
                oc_clfs.append(joblib.load(instr))
        return oc_clfs

    def annotate_data(self, base_filename):
        """ Load the one-class activity models and use them to label new data.
        Assume new data is input at the specified sample rate.
        """
        lines = list()
        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        annotated_datafile = os.path.join(self.conf.datapath, base_filename + '.anncombine')
        outfile = open(annotated_datafile, "w")

        # Load person stats.
        personfile = os.path.join(self.conf.datapath, base_filename + '.person')
        if not os.path.isfile(personfile):
            print(personfile, "does not exist, generating these stats")
            person.main(base_filename=base_filename,
                        cf=self.conf)
        person_stats = np.loadtxt(personfile, delimiter=',')
        oc_clusters = features.load_clusters(base_filename=base_filename,
                                             cf=self.conf)
        oc_clfs = self.load_models()

        count = 0
        for line in open(infile):
            count += 1
            lines.append(utils.process_entry(line))
            # Collect one set of sensor readings
            if (count % self.conf.numsensors) == 0 and count >= self.conf.windowsize:
                dt = self.read_sensors_from_window(lines)
                xpoint = features.create_point(self, dt, infile, person_stats, oc_clusters)
                xdata = [xpoint]
                newlabels = list()
                for i in range(self.conf.numactivities):
                    newlabel = oc_clfs[i].predict(xdata)[0]
                    if i == (self.conf.numactivities - 1):
                        newlabels.append(self.conf.activity_list.index(newlabel))
                    else:
                        newlabels.append(newlabel)
                self.output_combined_window(outfile, count, lines, newlabels)
                lines = lines[self.conf.numsensors:]  # delete one set of readings
        outfile.close()


def new_window(delta, gen, count, conf: config.Config):
    """ Determine if conditions are met to start a new window.
    """
    return ((delta.seconds > 2) or (gen == 1)) or \
           ((conf.annotate > 0) and ((count % conf.samplesize) == 0))


def end_window(label1, label2, count, conf: config.Config):
    """ Determine if conditions are met to end the window and add a labeled data
    point to the sample.
    """
    fullsize = conf.samplesize - 1
    return ((conf.annotate == 0) and (label1 != 'Ignore') and
            (label1 != 'None') and
            ((count % conf.samplesize) == (conf.samplesize - 1))) or \
           ((conf.annotate > 0) and ((count % conf.samplesize) == fullsize))


def extract_features(base_filename: str, oc: OC) -> (list, list):
    """ Extract a feature vector from the window of sensor data to use for
    classifying the data sequence.
    """
    xdata = list()
    ydata = list()
    personfile = os.path.join(oc.conf.datapath, base_filename + '.person')
    if not os.path.isfile(personfile):
        print(personfile, "does not exist, generating these stats")
        person.main(base_filename, oc.conf)
    person_stats = np.loadtxt(personfile, delimiter=',')
    datafile = os.path.join(oc.conf.datapath, base_filename + oc.conf.extension)
    oc_clusters = features.load_clusters(base_filename, oc.conf)

    infile = open(datafile, "r")  # process input file to create feature vector
    count = 0
    valid, date, time, f1, f2, v1, v2 = oc.location.read_entry(infile)
    prevdt = utils.get_datetime(date, time)
    gen = oc.resetvars()
    while valid:
        dt = utils.get_datetime(date, time)
        delta = dt - prevdt
        if new_window(delta, gen, count, oc.conf):  # start new window
            gen = oc.resetvars()
        pdt, v2, date, time, label1, label2 = oc.read_sensors(infile, v1)
        if end_window(label1, label2, count, oc.conf):
            gen = 1
            if oc.location.valid_location_data(oc.latitude, oc.longitude, oc.altitude):
                dt = utils.get_datetime(date, time)
                xpoint = features.create_point(oc, dt, base_filename, person_stats, oc_clusters)
                xdata.append(xpoint)  # learn the first and second activity classes
                ydata.append(label1)
                if label1 != label2 and oc.conf.annotate == 0:
                    xdata.append(xpoint)
                    ydata.append(label2)
        if not valid:
            prevdt = pdt
        else:
            prevdt = utils.get_datetime(date, time)
        count += 1
        valid, date, time, f1, f2, v1, v2 = oc.location.read_entry(infile)
    infile.close()
    return xdata, ydata


def parallel_extract_features(response_queue: Queue, base_filename: str, oc: OC):
    """ Handle multiple processes for extracting features.
    """
    print('extracting', base_filename)
    xdata, ydata = extract_features(base_filename, oc)
    print('extracted', base_filename)
    response_queue.put((xdata, ydata, base_filename, oc.location.locations,))
    return


def gather_sensor_features(files: list, oc: OC) -> (list, list):
    """ Fork a set of processes to gather statistical features, one for
    each input file.
    """
    """ Initializations done here because they need to happen whenever we are using
    this file, including in child processes.
    """
    xdata = list()
    ydata = list()

    feature_responses = Queue()  # create feature vectors from set of files
    feature_processes = list()
    for base_filename in files:
        feature_processes.append(Process(target=parallel_extract_features,
                                         args=(feature_responses, base_filename, oc,)))
    for i in range(len(feature_processes)):
        feature_processes[i].start()
    for i in range(len(feature_processes)):
        tmp_xdata, tmp_ydata, datafile, tmp_locations = feature_responses.get()
        xdata = xdata + tmp_xdata
        ydata = ydata + tmp_ydata
        oc.location.locations = oc.location.locations + tmp_locations
        print('{} of {} done: {}'.format(i + 1, len(feature_processes), datafile))
    for i in range(len(feature_processes)):
        feature_processes[i].join()
    return xdata, ydata


def oneclass(conf: config.Config):
    """ Perform one-class activity classification of input data.
    """
    oc = OC(conf=conf)

    if conf.locmodel == 1:
        oc.locclf = oc.location.load_location_model()
    if conf.translate:
        oc.aclass.read_activity_mappings_both()
        oc.location.read_location_mappings()
    oc.location.read_locations()

    if conf.mode == config.MODE_ANNOTATE_DATA:
        oc.annotate_data(base_filename=conf.files[0])
    else:
        # The 3 remaining modes all require x,y feature data, so we generate those here.
        xdata, ydata = gather_sensor_features(files=conf.files,
                                              oc=oc)

        if conf.mode == config.MODE_TEST_MODEL:
            oc.test_model(xdata=xdata,
                          ydata=ydata)
        elif conf.mode == config.MODE_TRAIN_MODEL:
            oc.train_model(xdata=xdata,
                           ydata=ydata)
        elif conf.mode == config.MODE_CROSS_VALIDATION:
            oc.cross_validate_model(xdata=xdata,
                                    ydata=ydata)

    if conf.gpsfeatures == 1 and conf.locmodel == 0:
        gps.update_locations(oc.location.locations, 'locations')
    return


def main():
    """ Initializations done here because they need to happen whenever we are using
    this file, including in child processes.
    """
    cf = config.Config(description='OneClass')
    cf.set_parameters()

    oneclass(conf=cf)
    return


if __name__ == "__main__":
    main()
