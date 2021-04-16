"""
Code used as base objects for derivative classes implementing individual model training and data
processing features.

Written by Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List

import config
import utils
from features import generate_features, calculate_space_features, calculate_time_features
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

        # Whether we just created a feature vector at the end of a window:
        self.generated_window = False

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

        self.resetvars()

        # Loop over all event rows in the input file:
        for event in in_data.rows_dict:
            # Get event's stamp and use that to compute delta since last event:
            dt = event[self.conf.stamp_field_name]

            # Set prevdt to this time if None (first event):
            if prevdt is None:
                prevdt = dt

            delta = dt - prevdt

            # Check if we should start a new window (reset window lists):
            if self.new_window(delta):
                self.resetvars()

            # Update the sensor values for this window:
            self.update_from_event(event)

            # Check if this should be the end of a window:
            if self.end_window(event):
                self.generated_window = True

                if self.should_create_feats_for_window(event):
                    xpoint = self.create_point(event)

                    self.process_window_feats(event, xpoint)

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

        # Reset the flag for when we just generated a window:
        self.generated_window = False

    def new_window(self, delta: timedelta) -> bool:
        """
        Determine if a new window should be created.

        TODO: Check on this code and clean up as applicable. Make sure it's doing what we expect
        Some things to check:
        - Should we use something else instead of `conf.annotate`? e.g. checking for annotate mode?

        Parameters
        ----------
        delta : timedelta
            The time since the last event

        Returns
        -------
        bool
            True if it is time to start a new window
        """

        return delta.seconds > 2 or self.generated_window or \
            (self.conf.annotate > 0 and self.count % self.conf.samplesize == 0)

    def end_window(self, latest_event: Dict[str, Union[datetime, float, str, None]]) -> bool:
        """
        Determine whether to end a window and calculate feature vector.

        By default, simply check if if we have `conf.samplesize` events in the window. Override
        this if you'd like further checks for window end.

        Parameters
        ----------
        latest_event : Dict[str, Union[datetime, float, str, None]]
            The latest event read from the file (which would form the end of the window)

        Returns
        -------
        bool
            True if we should end the window and calculate features now
        """

        return self.count % self.conf.samplesize == self.conf.samplesize - 1

    def update_from_event(self, event: Dict[str, Union[datetime, float, str, None]]):
        """
        Process a new sensor event to update state.

        By default, updates the sensor value lists with the new sensor values

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

    def update_location_range(self, value: float, datatype: str):
        """
        Update location range for the given datatype for this window.

        Parameters
        ----------
        value : float
            The value of the sensor
        datatype : str
            The type of data: either 'latitude' or 'longitude'
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

    def should_create_feats_for_window(self,
                                       latest_event: Dict[str, Union[datetime, float, str, None]]) \
            -> bool:
        """
        Whether we should create a feature vector for this window.

        By default, always return True. Override if you want to add further checks.

        Parameters
        ----------
        latest_event : Dict[str, Union[datetime, float, str, None]]
            The last event in the window

        Returns
        -------
        bool
            True if we should create a vector for this window.
        """

        return True

    @abstractmethod
    def create_point(self, latest_event: Dict[str, Union[datetime, float, str, None]]) \
            -> List[float]:
        """
        Create a feature vector (point) at the end of a window. You should implement this method
        in subclasses to derive the desired feature vector you want for that type of model.

        Helper methods `gen_motion_sensor_features()` and `gen_location_sensor_features()` can be
        used as default implementations of generating the typical motion sensor and location sensor
        based features used, respectively.

        Parameters
        ----------
        latest_event : Dict[str, Union[datetime, float, str, None]]
            The last event processed in the window

        Returns
        -------
        List[float]
            A feature vector created from the window
        """

        pass

    @staticmethod
    def default_motion_sensor_features(st: 'BaseDataProcessor', use_lowpass_filter: bool = False) \
            -> List[float]:
        """
        Helper default method.

        Generate motion-sensor-based features for a window. Uses the window values set on the
        `st` object to get sensor values.

        Parameters
        ----------
        st : BaseDataProcessor
            A data processor object that has window sensor values (e.g. yaw, pitch, etc) set on it
        use_lowpass_filter : bool, default False
            Whether to use a lowpass filter on the motion sensor vectors

        Returns
        -------
        List[float]
            A feature vector with the default motion sensor feature types
        """

        motion_feats = list()

        for i in [st.yaw, st.pitch, st.roll, st.rotx, st.roty,
                  st.rotz, st.accx, st.accy, st.accz, st.acctotal]:
            if use_lowpass_filter:
                i = utils.butter_lowpass_filter(i)

            motion_feats.extend(generate_features(x=i, cf=st.conf))

        return motion_feats

    @staticmethod
    def default_location_sensor_features(st: 'BaseDataProcessor') -> List[float]:
        """
        Helper default method.

        Generate location-sensor-based features for a window. Uses the window values set on the
        `st` object to get sensor values.

        Parameters
        ----------
        st : BaseDataProcessor
            A data processor object that has window sensor values (e.g. latitude, longitude, etc)
            set on it

        Returns
        -------
        List[float]
            A feature vector with the default location sensor feature types
        """

        loc_feats = list()

        for i in [st.latitude, st.longitude, st.altitude]:
            # Only include absolute features if enabled in config:
            loc_feats.extend(
                generate_features(x=i, cf=st.conf,
                                  include_absolute_features=st.conf.gen_gps_abs_stat_features)
            )

        for i in [st.course, st.speed, st.hacc, st.vacc]:
            loc_feats.extend(generate_features(x=i, cf=st.conf))

        return loc_feats

    @abstractmethod
    def process_window_feats(self,
                             latest_event: Dict[str, Union[datetime, float, str, None]],
                             feats: List[float]
                             ):
        """
        Called when we have a new window feature vector, so we can process it as needed.

        You MUST implement this method in a sub-class.

        This could be, for example, adding the values to `self.xdata` and adding a value to
        `self.ydata`. Alternatively, you could do something like write out annotated events to a
        file.

        Parameters
        ----------
        latest_event : Dict[str, Union[datetime, float, str, None]]
            The latest event in the window
        feats : List[float]
            The feature vector created for the window
        """

        pass
