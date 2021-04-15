"""
Code used as base objects for derivative classes implementing individual model training and data
processing features.

Written by Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""

from abc import ABC


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
