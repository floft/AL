"""
Script to annotate using both one- and multi-class models in the combined output CSV format for
used as input to the Digital Markers code.

This will be very similar to the outputs if `annotate != 1` in `al.py` and `oc.py`, but with both
the one-class labels from `oc.py` (first) and the multi-class activity index from `al.py` (last)
included in each output row.

This script does not include any functionality for training, testing, etc. It only supports
annotation in the above-described way. (Use the `al.py` and `oc.py` scripts to train the needed
models.)

Written by Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""
import collections
import os
from datetime import datetime
from typing import OrderedDict, Union, TextIO, Dict

import joblib
from sklearn.ensemble import RandomForestClassifier

from mobiledata import MobileData
from oc import OC


class DMAnnotator(OC):
    """
    Class for doing the DM annotation. Utilize the `OC` class for the base needed, but we will
    override to implement special model loading of both one- and multi-class models and forcing one
    output format.

    The base class's `annotate_data()` method should handle the underlying work needed for us once
    we pass it the right models.
    """

    def load_models(self) -> OrderedDict[str, RandomForestClassifier]:
        """
        Override model loading to load both one- and multi-class models. The one-class models
        are included first in the `OrderedDict` so the output format will have one-class labels
        first.
        """

        models = collections.OrderedDict()

        # Load one-class models (use the list of activities set in the config):
        for activity in self.conf.activities:
            model_filename = os.path.join(self.conf.modelpath, f'{activity}.pkl')

            models[activity] = joblib.load(model_filename)

        # Load the multi-class model:
        model_filename = os.path.join(self.conf.modelpath, 'model.pkl')
        models[DMAnnotator.multi_class_clf_field] = joblib.load(model_filename)

        return models

    def get_output_file_object(
            out_filename: str,
            fields: OrderedDict[str, str],
            output_dm_format: bool = False
    ) -> Union[TextIO, MobileData]:
        """
        Override to ALWAYS use a regular text object to write out in DM format.
        """

        return open(out_filename, 'w')  # regular file for writing DM CSV

    def write_event(
            self,
            out_data: Union[TextIO, MobileData],
            event: Dict[str, Union[datetime, float, str, None]],
            labels: OrderedDict[str, str],
            output_dm_format: bool = False
    ):
        """
        Override to ALWAYS write out the event in DM CSV format.
        """

        self.write_event_dm_format(out_data, event, labels)
