"""
Code used as base objects for derivative classes implementing individual model training and data
processing features.

Written by Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""

from abc import ABC

import config


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

        # Create an attribute for setting the classifier to use:
        # You should override this in an inherited initializer if you plan to use a classifier
        self.clf = None
