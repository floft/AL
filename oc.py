"""
Code for training and using one-class activity models, which predict a single class at a time. Based
on the code for multi-class model in `al.py`.

Written by Diane J. Cook and Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""
import collections
import os
from datetime import datetime
from typing import Dict, Union, Optional, List, OrderedDict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from al import AL, gather_features_by_file


class OC(AL):
    """
    Class to handle training/testing of one-class (OC) models.

    Uses base code from the `AL` class but overridden for one-class models.
    """

    def save_models(self, oc_models: Dict[str, RandomForestClassifier]):
        """
        Save the passed-in one-class models.

        Parameters
        ----------
        oc_models : Dict[str, RandomForestClassifier]
            The one-class models in a dictionary keyed by their activity names
        """

        print("Saving models...", end='')

        for activity, model in oc_models.items():
            model_filename = os.path.join(self.conf.modelpath, f'{activity}.pkl')

            joblib.dump(model, model_filename)

        print("done")

    def load_models(self) -> OrderedDict[str, RandomForestClassifier]:
        """
        Override model loading to load the one-class activity models.
        """

        models = collections.OrderedDict()

        # Use the list of activities set in the config:
        for activity in self.conf.activities:
            model_filename = os.path.join(self.conf.modelpath, f'{activity}.pkl')

            models[activity] = joblib.load(model_filename)

        return models

    def train_model(self, xdata: list, ydata: list):
        """
        Override training so that we train separate one-class models for each model specified
        in the configuration listing.

        We will train using the list of one-class activities provided on the config.
        """

        # Train the models, using the list of one-class activities from the config:
        oc_classifiers = self.train_models_for_data(xdata, ydata, self.conf.activities)

        # Save the models to disk:
        self.save_models(oc_classifiers)

    def train_models_for_data(
            self,
            xdata: List[List[float]],
            ydata: List[str],
            activities: List[str]
    ) -> OrderedDict[str, RandomForestClassifier]:
        """
        Train one-class activity models for the provided data and list of activities.

        We will train each model separately with an array of target binary values (1/0) indicating
        whether a given instance has a label that matches the given activity class for the model.

        Parameters
        ----------
        xdata : List[List[float]]
            The feature vectors to train with
        ydata : List[str]
            The original activity labels corresponding to the feature vectors
        activities : List[str]
            List of activities to train one-class models for

        Returns
        -------
        OrderedDict[str, RandomForestClassifier]
            A dictionary mapping the activities to their trained models, in the same order as input
        """

        classifiers = collections.OrderedDict()

        for activity in activities:
            print(f"Training one-class model for {activity}...", end='')
            classifiers[activity] = self.train_one_class_model(activity, xdata, ydata)
            print("done")

        return classifiers

    def train_one_class_model(
            self,
            activity: str,
            xdata: List[List[float]],
            ydata: List[str]
    ) -> RandomForestClassifier:
        """
        Train an individual one-class classifier for the given activity and x/y data.

        We will train the model with with an array of target binary values (1/0) indicating
        whether a given instance has a label that matches the given activity class for the model.

        Parameters
        ----------
        activity : str
            The activity name to train the model for
        xdata : List[List[float]]
            The feature vectors to train with
        ydata : List[str]
            The original activity labels corresponding to the feature vectors

        Returns
        -------
        RandomForestClassifier
            The trained one-class model for this activity
        """

        # Create a binary target vector with 1 if the label translates to this activity, 0 otherwise
        binary_labels = self.create_binary_target(activity, ydata)

        # Now train a model on the given data:
        clf = RandomForestClassifier(n_estimators=100,
                                     bootstrap=True,
                                     criterion="entropy",
                                     class_weight="balanced",
                                     max_depth=10,
                                     n_jobs=self.conf.oc_n_jobs)

        clf.fit(xdata, binary_labels)

        return clf

    def create_binary_target(self, activity: str, ydata: List[str]) -> np.ndarray:
        """
        Create a binary target vector (1/0 values) for the given activity based on the original
        labels in `ydata`.

        These values are determined by translating the original activity label (set on the `ydata`
        list) using the two activity mappings (previously loaded from oca.translate) and seeing if
        one of them matches the current activity.
        """

        # Create a binary target vector with 1 if the label translates to this activity, 0 otherwise
        binary_labels = np.zeros(len(ydata))

        for i, label in enumerate(ydata):
            # Translate the label if configured, otherwise just use the one original label:
            if self.conf.translate:
                label_translations = self.translate_label(label)
            else:
                label_translations = [label]

            # If the activity is one of the translations, mark this as a positive (1) example:
            if activity in label_translations:
                binary_labels[i] = 1

        return binary_labels

    def translate_label(self, label: str) -> List[str]:
        """
        Translate the given original label to the list of labels from the translation file.

        In the current case, we return the two labels returned from oca.translate.
        """

        return [
            self.aclass.map_activity_name(label),
            self.aclass.map_activity_name2(label)
        ]

    # TODO: Implement testing (and CV) at a later date
    # def test_models_on_data(
    #         self,
    #         models: OrderedDict[str, RandomForestClassifier],
    #         xdata: List[List[float]],
    #         ydata: List[str]
    # ):
    #     raise NotImplementedError()

    # def test_model_on_data(
    #         self,
    #         model_activity: str,
    #         model: RandomForestClassifier,
    #         xdata: List[List[float]],
    #         ydata: List[str]
    # ) -> Tuple[float, float, float, float, Tuple[List[Union[str, int]]]]:
    #     """
    #     Test the provided one-class model on the given data and return results.
    #
    #     For the model, the target values are a binary vector determined by translating the labels
    #     in `ydata` using the `oca.translate` translations (if configured), then setting the target
    #     to `1` if the translated names match the model activity, or otherwise `0`.
    #
    #     Parameters
    #     ----------
    #     model_activity : str
    #         The activity the model is trained for
    #     model : RandomForestClassifier
    #         The one-class activity model
    #     xdata : List[List[float]]
    #         The feature vectors to test on
    #     ydata : List[str]
    #         The original activity labels corresponding to the feature vectors
    #
    #     Returns
    #     -------
    #     Tuple[float, float, float, float, Tuple[List[Union[str, int]]]]]
    #         The following values resulting from the test:
    #          - Accuracy
    #          - Precision
    #          - Recall
    #          - F1 score
    #          - Tuple with two values:
    #            - List of ground-truth binary target values for this activity
    #            - List of predicted binary target values from the model
    #     """
    #
    #     # Create a binary target vector with 1 if the label translates to this activity, 0 otherwise
    #     binary_targets = self.create_binary_target(model_activity, ydata)
    #
    #     # Now make predictions on the feature vectors using the model:
    #     new_labels = model.predict(xdata)
    #
    #     # Calculate the different scores:
    #     ...

    def process_activity_label(self, event: Dict[str, Union[datetime, float, str, None]]) \
            -> Optional[str]:
        """
        Override the base activity label processing to NOT translate the activity name here.
        We will only replace spaces with underscores.

        We later will use the translations when determining individual one-class activity model
        positive/negative examples.
        """

        original_label = event[self.conf.label_field_name]

        # Return None if the original label is None:
        if original_label is None:
            return None

        # Replace spaces with underscores:
        cleaned_label = original_label.replace(' ', '_')

        return cleaned_label


def main():
    """
    Main function called when running this file as a script.

    Possible actions include:
     - Train one-class models for specified activities
     - Test trained one-class models
     - Cross-validation testing of one-class models
     - Leave-one-out testing of one-class models
     - Annotate data using one-class (and possibly multi-class) models
    """

    cf = config.Config(description='One-Class AL Activity Learning')
    cf.set_parameters()

    oc = OC(conf=cf)

    # Load the location model if we need to use it:
    if cf.locmodel == 1:
        oc.locclf = oc.location.load_location_model()

    # Load translations for activities and locations if needed:
    if cf.translate:
        oc.aclass.read_activity_mappings_both()  # oca.translate (map each activity to two possible)
        oc.location.read_location_mappings()

    # Read in existiong locations to pre-populate Location object:
    oc.location.read_locations()

    if cf.mode == config.MODE_ANNOTATE_DATA:
        oc.annotate_data(cf.files[0])
    elif cf.mode == config.MODE_LEAVE_ONE_OUT:
        raise NotImplementedError()
    else:
        # Train, test, CV modes require x,y feature data, so generate those here:
        data_by_file = gather_features_by_file(files=cf.files, al=oc)

        # For these options, we want to use all files' data at once, so combine them:
        xdata = list()
        ydata = list()

        for data in data_by_file.values():
            xdata = xdata + data[0]
            ydata = ydata + data[1]

        if cf.mode == config.MODE_TEST_MODEL:
            raise NotImplementedError()
        elif cf.mode == config.MODE_TRAIN_MODEL:
            oc.train_model(xdata, ydata)
        elif cf.mode == config.MODE_CROSS_VALIDATION:
            raise NotImplementedError()
        else:
            msg = f"The mode {cf.mode} is not supported by this script"
            raise ValueError(msg)


if __name__ == '__main__':
    main()
