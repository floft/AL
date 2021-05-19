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
import collections
import math
import os.path
from collections import deque
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Optional, Dict, Union, List, Tuple, TextIO, OrderedDict

import joblib
import numpy as np

import activity
import config
import features
import gps
import loc
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

    # Name to use for the multi-class classifier field in data and classifier dictionaries:
    multi_class_clf_field = 'multi_class_activity'

    # Names of sensor values to include in Digital Marker "combined CSV" format:
    dm_format_sensors = [
        'yaw',
        'pitch',
        'roll',
        'rotation_rate_x',
        'rotation_rate_y',
        'rotation_rate_z',
        'user_acceleration_x',
        'user_acceleration_y',
        'user_acceleration_z',
        'latitude',
        'longitude',
        'altitude',
        'course',
        'speed',
        'horizontal_accuracy',
        'vertical_accuracy'
    ]

    # Base date to use for "days" field for DM format:
    dm_base_date = datetime(year=2010, month=1, day=1, hour=0, minute=0, second=0)

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

    def load_models(self) -> OrderedDict[str, RandomForestClassifier]:
        """
        Load multi-class model and return it.

        Note that this function allows returning multiple models (for override in sub-classes), but
        we only return the one multi-class model.
        """

        modelfilename = os.path.join(self.conf.modelpath, 'model.pkl')
        models = collections.OrderedDict()

        with open(modelfilename, 'rb') as f:
            models[AL.multi_class_clf_field] = joblib.load(f)

        return models

    def test_model(self, xdata: list, ydata: list):
        """ Test an activity model on new data.
        """
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

    def annotate_data(self, base_filename: str):
        """
        Use an activity model to label new data. Assumes that the input data is at the sample rate
        specified in the config (and used to train the model).

        Parameters
        ----------
        base_filename : str
            The filename we wish to annotate
        """

        # Determine whether we'll write to regular CSV or the "combined CSV" format used for
        # Digital Markers
        # We will write as regular format if annotate == 1, otherwise use DM format
        output_dm_format = self.conf.annotate != 1

        # Keep track of the most recent window (samplesize rows) of data:
        window_events = deque(maxlen=self.conf.samplesize)

        # Check that the number of events between labels is not larger than the window size:
        if self.conf.ann_num_events_between_labels > self.conf.samplesize:
            msg = "# of events between labels ({}) larger than window size ({})".format(
                self.conf.ann_num_events_between_labels,
                self.conf.samplesize
            )
            warn(msg)
            warn("Some events will not be included in any feature vectors")

        # Track the events that will get the next label we generate:
        events_for_next_label = deque()

        # Tracking list of feature vectors and associated events used to form them:
        xdata = list()
        events_for_windows = list()

        # Load person stats.
        personfile = os.path.join(self.conf.datapath, base_filename + '.person')
        if not os.path.isfile(personfile):
            print(f"{personfile} does not exist, generating these stats")
            person.main(base_filename, self.conf)

        person_stats = np.loadtxt(personfile, delimiter=',')
        al_clusters = features.load_clusters(base_filename, self.conf)

        classifiers = self.load_models()

        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        annotated_datafile = os.path.join(self.conf.datapath, base_filename + '.ann')

        in_data = MobileData(infile, 'r')
        in_data.open()

        # Set up field info from the input file:
        fields = collections.OrderedDict(in_data.fields)

        # Add the classifier fields for labels:
        for clf_name in classifiers.keys():
            fields[clf_name] = 's'

        out_data = self.get_output_file_object(annotated_datafile, fields, output_dm_format)

        count = 0

        for event in in_data.rows_dict:
            count += 1

            window_events.append(event)

            # Add event to list that will get the next label:
            events_for_next_label.append(event)

            # Create feature vector and label starting with the first row where we have a full
            # window:
            if count >= self.conf.samplesize:
                xpoint = self.create_feature_vector(list(window_events), person_stats, al_clusters)

                xdata.append(xpoint)

                # Add the associated events that get this window's label to the list:
                if count == self.conf.samplesize:
                    # First window, so add all events in the window:
                    events_for_windows.append(list(window_events))
                else:
                    # Regular window, so just add the latest event:
                    events_for_windows.append([window_events[-1]])

                # Now actually do the predictions if there are enough vectors:
                if len(xdata) >= self.conf.num_wins_batch_predict:
                    # Now label the data with each classifier:
                    self.predict_and_write_events(xdata, events_for_windows, classifiers, out_data,
                                                  output_dm_format)

                    # Reset the lists:
                    xdata = list()
                    events_for_windows = list()

                    print(f"Wrote out up to {count} events")

        # Write out any final events:
        self.predict_and_write_events(xdata, events_for_windows, classifiers, out_data,
                                      output_dm_format)

        print(f"Wrote out up to {count} events")

        in_data.close()
        out_data.close()

        return

    @staticmethod
    def get_output_file_object(
            out_filename: str,
            fields: OrderedDict[str, str],
            output_dm_format: bool = False
    ) -> Union[TextIO, MobileData]:
        """
        Set up the output file object that will be used, based on the annotation setting.
        If `output_dm_format` is False (the default), we will output data in normal CSV format, so
        create a `MobileData` object for the output. Otherwise, if it's True, we will output in the
        Digital Marker "combined CSV" format, so use a normal file object.

        In either case, the object will be "opened" - so you will want to call `close()` on it
        later.

        Parameters
        ----------
        out_filename : str
            The name of the file we will write to
        fields : OrderedDict[str, str]
            The dictionary of fields to use if writing to standard CSV format
        output_dm_format : bool, default False
            Whether we'll output to a normal CSV file (if True) or the DM-input CSV format

        Returns
        -------
        Union[TextIO, MobileData]
            Either a MobileData object (if writing normal CSV) or a file object (for DM CSV)
        """

        if output_dm_format:
            return open(out_filename, 'w')  # regular file for writing DM CSV
        else:
            out_data = MobileData(out_filename, 'w')  # MobileData object for normal CSV output
            out_data.open()

            out_data.set_fields(fields)
            out_data.write_headers()

            return out_data

    def create_feature_vector(
            self,
            window_events: List[Dict[str, Union[datetime, float, str, None]]],
            person_stats: np.ndarray,
            al_clusters: List[List[float]]
    ) -> List[float]:
        """
        Create a feature vector for the events in the window.

        Parameters
        ----------
        window_events : List[Dict[str, Union[datetime, float, str, None]]]
            Window of events to create the feature vector for
        person_stats : np.ndarray
            The person stats for this data pulled from file
        al_clusters : List[List[float]]
            The cluster centers for the person clusters, pulled from file

        Returns
        -------
        List[float]
            The feature vector created for the window
        """

        # Reset the sensor lists, then populate with values from current window:
        self.resetvars()

        for win_event in window_events:
            self.update_sensors(win_event)

        # Get the timestamp of the most recent event:
        dt = window_events[-1][self.conf.stamp_field_name]

        return features.create_point(self, dt, person_stats, al_clusters)

    def predict_and_write_events(
            self,
            xdata: List[List[float]],
            events_for_windows: List[List[Dict[str, Union[datetime, float, str, None]]]],
            classifiers: OrderedDict[str, RandomForestClassifier],
            out_data: Union[TextIO, MobileData],
            output_dm_format: bool
    ):
        """
        Make predictions on the given feature vectors (`xdata`) and then write out the associated
        events along with their labels.

        Parameters
        ----------
        xdata : List[List[float]]
            List of feature vectors to label
        events_for_windows : List[List[Dict[str, Union[datetime, float, str, None]]]]
            List of sensor events associated with each feature vector/window, to write out
        classifiers : OrderedDict[str, RandomForestClassifier]
            The models to use for making predictions, in order they should appear in output
        out_data : Union[TextIO, MobileData]
            The file to write out data to
        output_dm_format : bool
            Whether to write the outputs in Digital Marker format
        """

        new_labels = collections.OrderedDict()

        for clf_name, clf in classifiers.items():
            new_labels[clf_name] = clf.predict(xdata)

        # Now write out the events from each prediction:
        for i, events_for_win in enumerate(events_for_windows):
            # Get the predictions for the given window index:
            labels_for_win = collections.OrderedDict({
                clf_name: predictions[i]
                for clf_name, predictions in new_labels.items()
            })

            for win_event in events_for_win:
                self.write_event(out_data, win_event, labels_for_win, output_dm_format)

    def write_event(
            self,
            out_data: Union[TextIO, MobileData],
            event: Dict[str, Union[datetime, float, str, None]],
            labels: OrderedDict[str, str],
            output_dm_format: bool = False
    ):
        """
        Write out an event to the output datafile with the specific label.

        If `output_dm_format` is False (the default), we will output data in normal CSV format.
        Otherwise, if it's True, we will output in the Digital Marker "combined CSV" format.

        We call the appropriate function to do the actual output based on this.

        Parameters
        ----------
        out_data : Union[TextIO, MobileData]
            The object to write output data to
        event : Dict[str, Union[datetime, float, str, None]]
            The event dictionaries for the sensor events in the window that was labeled
        labels : OrderedDict[str, str]
            The labels generated by models, keyed by their classifier name and in desired order
        output_dm_format : bool, default False
            Whether we'll output to a normal CSV file (if True) or DM "combined CSV" format
        """

        if output_dm_format:
            self.write_event_dm_format(out_data, event, labels)
        else:
            self.write_event_normal(out_data, event, labels)

    def write_event_normal(
            self,
            out_data: MobileData,
            event: Dict[str, Union[datetime, float, str, None]],
            labels: OrderedDict[str, str]
    ):
        """
        Write out an event to the output data file in normal CSV data format (one timestamp per
        line with all sensors listed), with the given label set.

        Parameters
        ----------
        out_data : MobileData
            The MobileData (file) object to write the output events to
        event : Dict[str, Union[datetime, float, str, None]]
            The event dictionaries for the sensor events in the window that was labeled
        labels : OrderedDict[str, str]
            The labels generated by models, keyed by their classifier name and in desired order
        """

        # Make a copy of the event and set the given labels:
        event_copy = dict(event)

        for clf_name, label in labels.items():
            event_copy[clf_name] = label

        # Write the event:
        out_data.write_row_dict(event_copy)

    def write_event_dm_format(
            self,
            out_data: TextIO,
            event: Dict[str, Union[datetime, float, str, None]],
            labels: OrderedDict[str, str]
    ):
        """
        Write out an event to the output data file in the Digital Marker "combined CSV" format.
        This has the following values on each line, separated by commas:
         - days since 2010-01-01
         - seconds since midnight on the day of the event
         - the 16 sensor values (per `AL.dm_format_sensors` value)
         - the activity labels, in order (usually one-class first, if included, then multi-class)
           (For the multi-class label, we convert it to an index based on the list in
           `conf.activity_list`, which should match what the classifier can output)

        Parameters
        ----------
        out_data : TextIO
            The MobileData (file) object to write the output events to
        event : Dict[str, Union[datetime, float, str, None]]
            The event dictionaries for the sensor events in the window that was labeled
        labels : OrderedDict[str, str]
            The labels generated by models, keyed by their classifier name and in desired order
        """

        # First set up the days/seconds values:
        event_stamp = event[self.conf.stamp_field_name]

        days = (event_stamp - AL.dm_base_date).days

        start_of_day = datetime(
            year=event_stamp.year,
            month=event_stamp.month,
            day=event_stamp.day,
            hour=0,
            minute=0,
            second=0
        )
        seconds = (event_stamp - start_of_day).seconds

        out_str = f'{days},{seconds}'

        # Now add the values for the sensors we want to include:
        for sensor in AL.dm_format_sensors:
            val = event[sensor] if event[sensor] is not None else 0.0  # set None to zero

            out_str += f',{val}'

        # Add the activity values:
        for clf_name, label in labels.items():
            # Convert the multi-class label to an integer:
            out_label = label
            if clf_name == AL.multi_class_clf_field:
                out_label = self.conf.activity_list.index(label)

            out_str += f',{out_label}'

        # Now write it to the file:
        out_str += '\n'
        out_data.write(out_str)

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
            count = 0

        # Update the sensor values for this window:
        al.update_sensors(event)

        # Get the processed label for this event:
        label = al.process_activity_label(event)

        if end_window(label, count, al.conf):
            gen = 1

            if al.location.valid_location_data(al.latitude, al.longitude, al.altitude):
                xpoint = features.create_point(al, dt, person_stats, al_clusters)

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


def leave_one_out(files: List[str], al: AL):
    """
    Run leave-one-out testing on the list of files. Will cycle through all files, leaving each one
    out and training on the rest, then testing on the left-out file and reporting results.

    Parameters
    ----------
    files : List[str]
        List of files to use. Should have at least two files
    al : AL
        The AL object to use for feature extraction and train/test. Should be pre-initialized.
    """

    if len(files) < 2:
        msg = "Need to have at least 2 files for leave-one-out testing"
        raise ValueError(msg)

    # Get the features for each file:
    file_data = gather_features_by_file(files, al)

    print("Collected data - starting leave-one-out")

    # Now loop through files and leave each one out and do train/test:
    for test_file in files:
        print(f"Leaving {test_file} out")

        # Get all other files except this one:
        train_files = [f for f in files if f != test_file]

        # Create the training data from other files:
        train_xdata = list()
        train_ydata = list()

        for train_file in train_files:
            train_xdata += file_data[train_file][0]
            train_ydata += file_data[train_file][1]

        # Now train the model:
        # TODO: Do we need to "reset" the classifier here?
        al.clf.fit(train_xdata, train_ydata)

        # Now test on the left-out file:
        print(f"Test results for left-out file {test_file}")

        test_xdata = file_data[test_file][0]
        test_ydata = file_data[test_file][1]

        al.test_model(test_xdata, test_ydata)


def gather_features_by_file(files: List[str], al: AL) -> Dict[str, Tuple[List, List]]:
    """
    Gather sensor features and labels from each of the files provided in sub-processes. This runs
    the extract_features() function in parallel across all files. We then collect the extracted
    feature vectors, labels, and locations from each file. The features and labels (`xdata` and
    `ydata`) are stored separately for each file, while the cached locations are added together
    and set on the al object.

    Parameters
    ----------
    files : List[str]
        List of files to use. Should have at least two files
    al : AL
        The AL object to use for feature extraction and train/test. Should be pre-initialized.

    Returns
    -------
    Dict[str, Tuple[List, List]]
        A dictionary mapping each file name to a tuple of `(xdata, ydata)` values for that file.
    """

    file_data = dict()

    feature_responses = Queue()  # create feature vectors from set of files
    feature_processes = list()

    for base_filename in files:
        feature_processes.append(Process(target=parallel_extract_features,
                                         args=(feature_responses, base_filename, al,)))

    for i in range(len(feature_processes)):
        feature_processes[i].start()

    for i in range(len(feature_processes)):
        file_xdata, file_ydata, datafile, tmp_locations = feature_responses.get()

        file_data[datafile] = (file_xdata, file_ydata)

        al.location.locations = al.location.locations + tmp_locations

        print('{} of {} done: {}'.format(i + 1, len(feature_processes), datafile))

    for i in range(len(feature_processes)):
        feature_processes[i].join()

    return file_data


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

    if cf.mode == config.MODE_ANNOTATE_DATA:
        al.annotate_data(base_filename=cf.files[0])
    elif cf.mode == config.MODE_LEAVE_ONE_OUT:
        # Do special leave-one-out train/test:
        leave_one_out(files=cf.files, al=al)
    else:
        # The 3 remaining modes all require x,y feature data, so we generate those here.
        data_by_file = gather_features_by_file(files=cf.files, al=al)

        # For these options, we want to use all files' data at once, so combine them:
        xdata = list()
        ydata = list()

        for data in data_by_file.values():
            xdata = xdata + data[0]
            ydata = ydata + data[1]

        if cf.mode == config.MODE_TEST_MODEL:
            # Test our pre-trained model.
            al.clf = al.load_models()[AL.multi_class_clf_field]  # load the multi-class act model
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
