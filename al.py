#!/usr/bin/python

# python al.py <data_file>+
#
# Performs activity learning on the given data files and outputs either the
# learned model, or the confusion matrices and accuracy for a 3-fold
# cross-validation test.
#
# This program assumes that data are continuous. If not, missing data
# should be imputed before running the program.
#
# If annotate==0, then the syntax for the program is python al.py file[s].
# In this use case, the program either trains and stores a model from the
# labeled data (cv=0) or trains and tests models from the labeled data using
# k-fold cross validation (where k=cv).
#
# If annotate==1, then the synax is python al.py datafile modelfile.
# In this use case, the program loads a pre-trained model (from modelfile) and
# uses is to assign activity labels to data (from datafile), storing the results
# in a new file.
#
# If annotate==2, the use case is similar to annotate=1. The difference is that
# the output contains one line per time step with date, time, all of the sensor
# readings, and the activity label (instead of one line per sensor reading).
#
# Written by Diane J. Cook, Washington State University.
#
# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.
import math
import os.path
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Optional, Dict, Union

import joblib
import numpy as np

import activity
import config
import features
import gps
import loc
import oc
import person
import utils
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from mobiledata import MobileData


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class AL:

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
                                          n_jobs=self.conf.al_n_jobs)
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
        """ Perform k-fold cross validation on the model.
        """
        activity_list = sorted(set(ydata))
        print('Activities:', ' '.join(activity_list))
        print('   (i,j) means true label is i, predicted as j')
        for i in range(self.conf.cv):
            numright = total = 0
            xtrain, xtest, ytrain, ytest = train_test_split(xdata,
                                                            ydata,
                                                            test_size=0.33,
                                                            random_state=i)
            self.clf.fit(xtrain, ytrain)
            newlabels = self.clf.predict(xtest)
            matrix = confusion_matrix(ytest, newlabels, labels=activity_list)
            print('fold', i + 1, '\n', matrix)
            for j in range(len(ytest)):
                if newlabels[j] == ytest[j]:
                    numright += 1
                total += 1
            print('accuracy', float(numright) / float(total))
        return

    def train_model(self, xdata: list, ydata: list):
        """ Train an activity model from features and labels extracted from
        training data.
        """
        self.clf.fit(xdata, ydata)
        outstr = self.conf.modelpath + "model.pkl"
        joblib.dump(self.clf, outstr)
        return

    def test_model(self, xdata: list, ydata: list):
        """ Test an activity model on new data.
        """
        modelfilename = self.conf.modelpath + "model.pkl"
        with open(modelfilename, 'rb') as f:
            self.clf = joblib.load(f)
        activity_list = sorted(set(ydata))
        print('Activities:', ' '.join(activity_list))
        print('   (i,j) means true label is i, predicted as j')
        numright = total = 0
        newlabels = self.clf.predict(xdata)
        matrix = confusion_matrix(ydata, newlabels, labels=activity_list)
        for j in range(len(ydata)):
            if newlabels[j] == ydata[j]:
                numright += 1
            total += 1
        print('test accuracy', float(numright) / float(total), '\n', matrix)
        print(classification_report(ydata, newlabels))
        return

    def output_window(self, outfile, count, lines, newlabel):
        """ Write a single window of annotated sensor readings to a file.
        """
        if count == self.conf.windowsize:  # Process beginning before full window
            start = 0
        else:
            start = self.conf.windowsize - self.conf.numsensors
        for i in range(start, self.conf.windowsize):
            outstr = '{} {} {} {} {} {}\n'.format(lines[i][0],
                                                  lines[i][1],
                                                  lines[i][2],
                                                  lines[i][3],
                                                  lines[i][4],
                                                  newlabel)
            outfile.write(outstr)
        return

    def output_combined_window(self, outfile, count, lines, newlabel):
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
                    outstr += ',' + newlabel + '\n'
                    outfile.write(outstr)
        return

    def annotate_data(self, base_filename: str):
        """ Load the activity model and use it to label new data.
        Assume new data is input at the specified sample rate.
        """
        lines = list()

        # Load person stats.
        personfile = os.path.join(self.conf.datapath, base_filename + '.person')
        if not os.path.isfile(personfile):
            print(personfile, "does not exist, generating these stats")
            person.main(base_filename=base_filename,
                        cf=self.conf)
        person_stats = np.loadtxt(personfile, delimiter=',')
        al_clusters = features.load_clusters(base_filename=base_filename,
                                             cf=self.conf)

        # Load model.
        instr = os.path.join(self.conf.modelpath, 'model.pkl')
        if not os.path.isfile(instr):
            print(instr, "model file does not exist")
            exit()
        clf = joblib.load(instr)

        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        annotated_datafile = os.path.join(self.conf.datapath, base_filename + '.ann')
        outfile = open(annotated_datafile, "w")
        count = 0
        for line in open(infile):
            count += 1
            lines.append(utils.process_entry(line))
            # Collect one set of sensor readings
            if (count % self.conf.numsensors) == 0 and count >= self.conf.windowsize:
                dt = self.read_sensors_from_window(lines)
                xpoint = features.create_point(self, dt, infile, person_stats, al_clusters)
                xdata = [xpoint]
                newlabel = clf.predict(xdata)[0]
                if self.conf.annotate == 1:
                    self.output_window(outfile, count, lines, newlabel)
                else:
                    self.output_combined_window(outfile, count, lines, newlabel)
                lines = lines[self.conf.numsensors:]  # delete one set of readings
        outfile.close()
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
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.pitch.append(utils.clean(float(v1), -5.0, 5.0))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.roll.append(utils.clean(float(v1), -5.0, 5.0))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.rotx.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.roty.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.rotz.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accx.append(v1)
        temp = v1 * v1
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accy.append(v1)
        temp += v1 * v1
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        v1 = utils.clean(float(v1), -1.0, 1.0)
        self.accz.append(v1)
        temp += v1 * v1
        self.acctotal.append(np.sqrt(temp))  # compute combined acceleration

        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.latitude.append(float(v1))
        self.update_location_range(float(v1), datatype="latitude")
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.longitude.append(float(v1))
        self.update_location_range(float(v1), datatype="longitude")
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.altitude.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.course.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.speed.append(float(v1))
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.hacc.append(float(v1))
        pdt = utils.get_datetime(date, time)
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        self.vacc.append(float(v1))
        return pdt, v2, date, time

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

    def read_entry(self, infile):
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
                if self.conf.translate:
                    x[5] = self.aclass.map_activity_name(x[5])
                return True, x[0], x[1], x[2], x[3], x[4], x[5]
        except:
            return False, None, None, None, None, None, None

    def process_activity_label(self, event: Dict[str, Union[datetime, float, str, None]]) \
            -> Optional[str]:
        """
        Process the activity label from the given event. This involves replacing spaces with
        underscores, and translating the activity name if `translate` is set to True in the config.

        Parameters
        ----------
        event : Dict[str, Union[datetime, float, str, None]]
            An event dictionary with fields mapped to their respective values
            Normally these would be output from the CSV Data Layer

        Returns
        -------
        Optional[str]
            The cleaned and (possibly) translated event name, or None if the original was None
        """

        original_label = event[self.conf.label_field_name]

        # Return None if the original label is None:
        if original_label is None:
            return None

        # Replace spaces with underscores:
        cleaned_label = original_label.replace(' ', '_')

        # Translate the label if configured:
        if self.conf.translate:
            cleaned_label = self.aclass.map_activity_name(cleaned_label)

        return cleaned_label


def new_window(delta, gen, count, conf: config.Config):
    """ Determine if conditions are met to start a new window.
    """
    return ((delta.seconds > 2) or (gen == 1)) or \
           ((conf.annotate > 0) and ((count % conf.samplesize) == 0))


def end_window(label: Optional[str], count: int, conf: config.Config) -> bool:
    """
    Check whether conditions are met to end the window and generate a feature vector.

    Parameters
    ----------
    label : Optional[str]
        The processed (and translated, if needed) activity label for the current event
    count : int
        Number of events processed overall
    conf : config.Config
        The configuration to check things against

    Returns
    -------
    bool
        True if we should end the window and create a feature vector/data point
    """

    fullsize = conf.samplesize - 1

    if conf.annotate == 0:
        return label is not None and label != 'None' and label != 'Ignore' \
               and count % conf.samplesize == conf.samplesize - 1
    elif conf.annotate > 0:
        return count % conf.samplesize == fullsize
    else:
        return False


def extract_features(base_filename: str, al: AL) -> (list, list):
    """
    Extract feature vectors from the provided file using the given AL object, which should at the
    least have its config and initialization set up already.

    Will read in the file indicated by the base name, parsing each event and creating windows.
    When a window is complete, extract features from it for an activity (AL) model, and add the
    feature vector and ground-truth activity label to `xdata` and `ydata`, respectively.

    Parameters
    ----------
    base_filename : str
        The "base" filename for the file to read - will be prepended with the data path and have the
        data file extension added to the end (from config)
    al : AL
        An `AL` object which has config and initialization done on it

    Returns
    -------
    (list, list)
        `xdata` and `ydata` containing feature vectors and ground-truth activity labels,
        respectively, for extracted windows.
    """

    xdata = list()
    ydata = list()

    # Load person stats and clusters:
    # If the .person file for this data does not exist, create it first
    personfile = os.path.join(al.conf.datapath, base_filename + '.person')
    if not os.path.isfile(personfile):
        print(f"{personfile} does not exist, generating these stats")
        person.main(base_filename, al.conf)

    person_stats = np.loadtxt(personfile, delimiter=',')  # person statistics
    al_clusters = features.load_clusters(base_filename, al.conf)

    # Load the data file through the CSV data layer:
    datafile = os.path.join(al.conf.datapath, base_filename + al.conf.extension)

    in_data = MobileData(datafile, 'r')
    in_data.open()

    count = 0

    prevdt = None  # type: Optional[datetime]

    al.resetvars()
    gen = 0

    # Loop over all event rows in the input files:
    for event in in_data.rows_dict:
        # Get the event's stamp:
        dt = event[al.conf.stamp_field_name]

        # Set prevdt to this time if None (first event):
        if prevdt is None:
            prevdt = dt

        delta = dt - prevdt

        if new_window(delta, gen, count, al.conf):  # start new window
            al.resetvars()
            gen = 0

        # Update the sensor values for this window:
        al.update_sensors(event)

        # Get the processed label for this event:
        label = al.process_activity_label(event)

        if end_window(label, count, al.conf):
            gen = 1

            if al.location.valid_location_data(al.latitude, al.longitude, al.altitude):
                xpoint = features.create_point(al, dt, base_filename, person_stats, al_clusters)

                xdata.append(xpoint)
                ydata.append(label)

        prevdt = dt

        count += 1

    in_data.close()

    return xdata, ydata


def parallel_extract_features(response_queue: Queue, base_filename: str, al: AL):
    """ Handle multiple processes for extracting features.
    """
    print('extracting', base_filename)
    xdata, ydata = extract_features(base_filename, al)
    print('extracted', base_filename)
    response_queue.put((xdata, ydata, base_filename, al.location.locations,))
    return


def gather_sensor_features(files: list, al: AL) -> (list, list):
    """ Fork a set of processes to gather statistical features, one for
    each input file.
    """
    xdata = list()
    ydata = list()
    feature_responses = Queue()  # create feature vectors from set of files
    feature_processes = list()
    for base_filename in files:
        feature_processes.append(Process(target=parallel_extract_features,
                                         args=(feature_responses, base_filename, al,)))
    for i in range(len(feature_processes)):
        feature_processes[i].start()
    for i in range(len(feature_processes)):
        tmp_xdata, tmp_ydata, datafile, tmp_locations = feature_responses.get()
        xdata = xdata + tmp_xdata
        ydata = ydata + tmp_ydata
        al.location.locations = al.location.locations + tmp_locations
        print('{} of {} done: {}'.format(i + 1, len(feature_processes), datafile))
    for i in range(len(feature_processes)):
        feature_processes[i].join()
    return xdata, ydata


def main():
    """ Initializations done here because they need to happen whenever we are using
    this file, including in child processes.

    Possible Actions:
        - Annotate single data file using a existing trained model.
        - Train single AL model on full data.
        - Train OC models on full data.
        - Test single AL model on full data.
        - Test OC models on full data.
    """
    cf = config.Config(description='AL Activity Learning')
    cf.set_parameters()
    al = AL(conf=cf)
    if cf.locmodel == 1:
        al.locclf = al.location.load_location_model()
    if cf.translate:
        al.aclass.read_activity_mappings()
        al.location.read_location_mappings()
    al.location.read_locations()

    if cf.oneclass:
        # Run our modes on the oneclass models instead of AL.
        oc.oneclass(conf=cf)
    elif cf.mode == config.MODE_ANNOTATE_DATA:
        al.annotate_data(base_filename=cf.files[0])
    else:
        # The 3 remaining modes all require x,y feature data, so we generate those here.
        xdata, ydata = gather_sensor_features(files=cf.files,
                                              al=al)

        if cf.mode == config.MODE_TEST_MODEL:
            # Test our pre-trained model.
            al.test_model(xdata=xdata,
                          ydata=ydata)
        elif cf.mode == config.MODE_TRAIN_MODEL:
            # Train our model.
            al.train_model(xdata=xdata,
                           ydata=ydata)
        elif cf.mode == config.MODE_CROSS_VALIDATION:
            # Perform cross validation with our model.
            al.cross_validate_model(xdata=xdata,
                                    ydata=ydata)

    if cf.gpsfeatures == 1 and cf.locmodel == 0:
        # Go ahead and save any new gps locations if we used gps features and did not use
        # a location model.
        gps.update_locations(al.location.locations, 'locations')


if __name__ == "__main__":
    main()
