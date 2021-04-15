"""
Code used as base objects for derivative classes implementing individual model training and data
processing features.

Written by Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""
import os
from abc import ABC
from datetime import datetime, timedelta
from typing import Optional

import config
from mobiledata import MobileData


class BaseDataProcessor(ABC):
    """
    Class used as a basis for other classes that implement data processing and model train/test
    operations. This class provides base implementations of most of these features, and should be
    inherited by classes implementing similar functionality. In these subclasses you can then
    override the base implementation as needed for customizations on that class.

    Note that this class is an abstract base class (ABC), meaning you cannot instantiate it.
    Rather, you should subclass it and then instantiate that class to use.

    Functionality provided by this class:
    - Initialization and setup of object properties such as sensor lists and setting config
    - An `extract_features()` method which will read through a data file and process out its events
      to create feature vectors. This method defines default implementation, which can then be
      overridden as needed, including:
      - A `before_extract()` method for doing things at the start of the extraction
      - `new_window()` and `reset_window()` methods for defining when to have a new window, etc
      - A `update_from_event()` method for handling updating sensor tracking based on a new event
      TODO: Add other pieces here
    """

    def __init__(self, conf: config.Config):
        """
        Default initializer. This will set the given config for the class, and also set up default
        sensor value lists for tracking sensors.

        Override this method (while calling it as a super) to add additional functionality.

        Parameters
        ----------
        conf : config.Config
            An AL Config object that has been loaded with the configuration to use
            Gets attached to this object
        """

        self.conf = conf

        # For holding feature vectors and ground-truth labels:
        self.xdata = list()
        self.ydata = list()

        # Create default lists for storing sensor values in windows:
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

        # Set up default min/max lat/long values for use:
        self.minlat = 90.0
        self.maxlat = -90.0
        self.minlong = 180.0
        self.maxlong = -180.0

        # Count for feature extraction:
        self.count = 0

        # Create an attribute for setting the classifier to use:
        # You should override this in an inherited initializer if you plan to use a classifier
        self.clf = None

    def extract_features(self, base_filename: str):
        """
        Process an input file row-by-row to extract features and process events.
        This is used as the basis for both extracting feature vectors for train/test, as well as
        for annotation.

        The method will start by opening the data file and doing other setup. Then, it reads in the
        file row-by-row.

        For each row, it first checks if the window should be reset by checking the `new_window()`
        method. Then, it calls `update_from_event()` with the new event. In the default
        implementation, this adds the event's sensor values to the sensor lists for the window.
        Then, the `end_window()` method is called to check and see if we have reached the end of a
        window.

        If so, we then proceed to create the feature vector for the window using the
        `create_point()` method. This is a base implementation, which will create the feature
        vector with most features in it. You should override this instead if you want a different
        set of features. (Sub-functions for each feature "group" are also provided to make this
        easier to change.)

        Then, the `new_vector()` method is called to let you handle what to do with the new feature
        vector. For example, you could add the vector to `self.xdata` and assign an appropriate
        `self.ydata` value.

        Parameters
        ----------
        base_filename : str
            The base filename (before datapath and extension are added) to process
        """

        infile = os.path.join(self.conf.datapath, base_filename + self.conf.extension)
        in_data = MobileData(infile, 'r')
        in_data.open()

        self.count = 0

        prevdt = None  # type: Optional[datetime]

        gen = self.resetvars()

        # Loop over all event rows in the input file:
        for event in in_data.rows_dict:
            # Get event's stamp and use that to compute delta since last event:
            dt = event[self.conf.stamp_field_name]

            # Set prevdt to this time if None (first event):
            if prevdt is None:
                prevdt = dt

            delta = dt - prevdt

            if self.new_window(delta, gen):  # start new window
                gen = self.resetvars()

            # Update the sensor values for this window:
            self.update_sensors(event)

            if (self.count % self.conf.samplesize) == (self.conf.samplesize - 1):  # end of window
                xpoint = list()
                gen = 1

                if self.valid_location_data(self.latitude, self.longitude, self.altitude):
                    for i in [self.yaw, self.pitch, self.roll, self.rotx, self.roty,
                              self.rotz, self.accx, self.accy, self.accz, self.acctotal]:
                        xpoint.extend(features.generate_features(i, self.conf))

                    for i in [self.latitude, self.longitude, self.altitude]:
                        # Only include absolute features if enabled in config:
                        xpoint.extend(features.generate_features(i, self.conf,
                                                                 include_absolute_features=self.conf.gen_gps_abs_stat_features))

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

            self.count += 1

            if (self.count % 100000) == 0:
                print('count', self.count)

        in_data.close()

    def resetvars(self):
        """
        Clear sensor value arrays and min/max lat/long values for a new window.
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

    def new_window(self, delta: timedelta, gen: int):
        """
        Determine if a new window should be created.

        TODO: Check on this code and clean up as applicable. Make sure it's doing what we expect
        Some things to check:
        - Switch `gen` to a bool
        - Should we use something else instead of `conf.annotate`? e.g. checking for annotate mode?

        Parameters
        ----------
        delta : timedelta
            The time since the last event
        gen : int
            If 1, indicates we just finished a window, so we should restart now

        Returns
        -------
        bool
            True if it is time to start a new window
        """

        return delta.seconds > 2 or gen == 1 or \
            (self.conf.annotate > 0 and self.count % self.conf.samplesize == 0)
