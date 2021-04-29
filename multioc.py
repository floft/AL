"""
`python multioc.py <config_parameters> <data_file>+`

Perform activity learning on the given data files with provided config parameters. In this case,
the models trained include a group of one-class classifiers, each trained for one of the activities
in the original data labels. These one-class classifiers are then used to label the feature vectors
and their outputs are appended to the end of each vector as additional features.

These new enhanced feature vectors are then combined with the translated activity names if the
translation is enabled. (In either case, instances with labels of 'Other' or 'Ignore' are dropped.)
This data is then used to train a multi-class classifier.

When testing, the one-class models are used to first label each feature vector and append their
value to the vector, then passed through the multi-class classifier.

(Note that only train, test, and loo modes are currently supported.)
"""
import collections
from datetime import datetime
from typing import Dict, Union, Optional, List, Tuple, OrderedDict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import al
import config


class MultiOC(al.AL):
    """
    Main class to run "multi one-class" activity modeling. Inherits from AL so that we can use most
    of its functions.
    """

    def process_activity_label(self, event: Dict[str, Union[datetime, float, str, None]]) \
            -> Optional[str]:
        """
        Override the original label processing to just convert spaces to underscores. Don't do
        translations.

        This will result in the `ydata` labels of the feature extraction being original activity
        names.

        Parameters
        ----------
        event : Dict[str, Union[datetime, float, str, None]]
            An event dictionary with fields mapped to their respective values
            Normally these would be output from the CSV Data Layer

        Returns
        -------
        Optional[str]
            The cleaned but not translated event name, or None if the original was None
        """

        original_label = event[self.conf.label_field_name]

        # Return None if the original label is None:
        if original_label is None:
            return None

        # Replace spaces with underscores:
        return original_label.replace(' ', '_')

    @staticmethod
    def get_oneclass_labels(ydata: List[str]) -> List[str]:
        """
        Get the activity labels from the ydata that we will use for one-class classifiers.
        That is, all labels except 'Other'.
        """

        # Reduce down to set of unique names:
        all_acts = set(ydata)

        # Return all of them except 'Other', sorted alphabetically:
        return sorted([a for a in all_acts if a != 'Other'])

    def train_model(self, xdata: list, ydata: list):
        """
        Train one-class and multi-class activity models on the provided features and original
        labels.

        Override the base class os we can do the special one-class training here.

        Parameters
        ----------
        xdata : list
            Input (original) feature vectors to train with
        ydata : list
            Input (original) activity labels corresponding to the feature vectors
        """

        # Get the activities for one-class classifiers:
        oc_activities = self.get_oneclass_labels(ydata)

        # Train the models:
        oneclass_models, multiclass_model = self.train_models_for_data(xdata, ydata, oc_activities)

        # Dump one-class models:
        for activity, oc_clf in oneclass_models.items():
            oc_outstr = self.conf.modelpath + activity + ".pkl"
            joblib.dump(oc_clf, oc_outstr)

        # Dump the multi-class model:
        outstr = self.conf.modelpath + "model.pkl"
        joblib.dump(multiclass_model, outstr)

    def train_models_for_data(
            self,
            xdata: List[List[float]],
            ydata: List[str],
            activities: List[str]
    ) -> Tuple[OrderedDict[str, RandomForestClassifier], RandomForestClassifier]:
        """
        Train one-class and then a multi-class classifier on the provided data. The `xdata` should
        be original feature vectors, and the `ydata` the original (untranslated) labels.

        This method trains a one-class classifier for each of the provided activities (which
        should normally not include 'Other'). Each classifier will be trained with a y-value vector
        with value `1` if the original label matches the one-class activity, or else `0`.

        Then, depending on the status of the config's `multioc_ground_truth_train` parameter,
        append additional features to the vector, one for each one-class activity, as to whether
        that instance has that one-class activity detected:
         - If `multioc_ground_truth_train` is True, use the ground-truth label on the instance (i.e.
           the one-class feature matching the instance label gets `1`, `0` for others)
         - If False, use the output of the corresponding one-class classifier on the feature vector.

        Translate the activity labels (if configured to do so), and then drop all instances where
        the label is 'Ignore'. Then train the multi-class activity classifier on the expanded
        feature vectors and these new labels.

        Parameters
        ----------
        xdata : List[List[float]]
            Input (original) feature vectors to train with
        ydata : List[str]
            Input (original) activity labels corresponding to the feature vectors
        activities : List[str]
            List of activities to train one-class classifiers for

        Returns
        -------
        Tuple[OrderedDict[str, RandomForestClassifier], RandomForestClassifier]
            An ordered dictionary mapping activity names to one-class classifiers (in order that
            features should be applied to existing vectors), and the multi-class classifier.
        """

        oc_classifiers = collections.OrderedDict()

        # Train the one-class classifier for each activity:
        for activity in activities:
            print(f"Training one-class classifier for {activity}...", end='')
            act_classifier = self.train_one_class(activity, xdata, ydata)

            oc_classifiers[activity] = act_classifier
            print("done")

        # Add the one-class features to the end of the original feature vectors:
        print("Adding one-class features to feature vectors")

        if self.conf.multioc_ground_truth_train:
            print("Using ground-truth label features...")
        else:
            print("Using predictions of one-class classifiers...")

        oc_xdata = self.add_oneclass_features(oc_classifiers, xdata, ydata)

        print("Done")

        # Now translate the activities and remove 'Ignore' ones:
        print("Translating labels and filtering for 'Ignore' and 'None'...", end='')

        new_xdata = list()
        new_ydata = list()

        for i, orig_label in enumerate(ydata):
            new_label = orig_label

            if self.conf.translate:
                new_label = self.aclass.map_activity_name(orig_label)

            # Only add instances that don't have 'Ignore' or 'None' for labels:
            if new_label != 'Ignore' and new_label != 'None':
                new_xdata.append(oc_xdata[i])
                new_ydata.append(new_label)

        print("done")

        # Now train the multi-class classifier on the new data:
        # (self.clf is the AL multi-class classifier configured instance)
        print("Training multi-class classifier...", end='')

        self.clf.fit(new_xdata, new_ydata)

        print("done")

        return oc_classifiers, self.clf

    def train_one_class(self, activity: str, xdata: List[List[float]], ydata: List[str]) \
            -> RandomForestClassifier:
        """
        Train the one-class classifier for the given activity based on the input features and
        original labels (`ydata`). We translate the original labels to a binary vector that is `1`
        whenever the label matches our given activity, then train the model.

        Parameters
        ----------
        activity : str
            The activity to train the classifier for.
        xdata : List[List[float]]
            Input (original) feature vectors to train with
        ydata : List[str]
            Input (original) activity labels corresponding to the feature vectors

        Returns
        -------
        RandomForestClassifier
            The trained one-class classifier.
        """

        # Create label vector where value is 0 unless the label == activity:
        binary_labels = np.zeros(len(ydata))

        for i, label in enumerate(ydata):
            if label == activity:
                binary_labels[i] = 1

        # Now train the classifier for these values with original features:
        clf = RandomForestClassifier(n_estimators=100,
                                     bootstrap=True,
                                     criterion="entropy",
                                     class_weight="balanced",
                                     max_depth=10,
                                     n_jobs=self.conf.oc_n_jobs)

        clf.fit(xdata, binary_labels)

        return clf

    def add_oneclass_features(
            self,
            oc_models: OrderedDict[str, RandomForestClassifier],
            xdata: List[List[float]],
            ydata: List[str]
    ) -> List[List[float]]:
        """
        Add oneclass activity features to the ends of existing feature vectors. These will be
        additional features that indicate whether each instance is an instance of the one-class
        activity or not (in order of the keys in the `oc_models` input).

        The way these features are formed depends on the value of the config's
        `multioc_ground_truth_train` value:
         - If True: Use ground-truth features - the value is `1` if the `ydata` label for that
         instance matches the activity, otherwise `0`
         - If False: Have each one-class classifier predict for the given feature vector and use
         those values.

        Parameters
        ----------
        oc_models : OrderedDict[str, RandomForestClassifier]
            The one-class classifiers keyed to their activity (in order we want features to be)
        xdata : List[List[float]]
            Input (original) feature vectors to train with
        ydata : List[str]
            Input (original) activity labels corresponding to the feature vectors

        Returns
        -------
        List[List[float]]
            The new feature vectors with one-class features appended
        """

        xdata_with_oc = list()

        for i, xpoint in enumerate(xdata):
            if self.conf.multioc_ground_truth_train:
                # We want to use ground-truth labels to create one-class features:
                new_xpoint = list(xpoint)

                # Set value to 1 for activity that matches the label:
                oc_feats = [1 if act == ydata[i] else 0 for act in oc_models.keys()]
                new_xpoint.extend(oc_feats)
            else:
                # We want to predict using each of the one-class classifiers:
                new_xpoint = self.add_oc_predictions(xpoint, oc_models)

            xdata_with_oc.append(new_xpoint)

        return xdata_with_oc

    @staticmethod
    def add_oc_predictions(
            xpoint: List[float],
            oc_models: OrderedDict[str, RandomForestClassifier]
    ) -> List[float]:
        """
        Add predictions from the one-class classifiers (in order from the dict) to the feature
        vector, with each one predicting on the original input feature vector.

        Parameters
        ----------
        xpoint : List[float]
            Feature vector to use for predictions and to extend
        oc_models : OrderedDict[str, RandomForestClassifier]
            Ordered dictionary of the one-class classifiers (in order we want to use them)

        Returns
        -------
        List[float]
            A copy of the input feature vector with the oc features appended
        """

        new_xpoint = list(xpoint)

        for oc_clf in oc_models.values():
            prediction = oc_clf.predict([xpoint])
            new_xpoint.append(prediction[0])

        return new_xpoint

    @staticmethod
    def test_models_for_data(
            oneclass_models: OrderedDict[str, RandomForestClassifier],
            multiclass_model: RandomForestClassifier,
            xdata: List[List[float]]
    ) -> List[str]:
        """
        Test multi-one-class models on the given data.

        First use the one-class models to predict each feature vector and add the outputs as
        "one-class features" to the vectors. Then use the multi-class classifier to make a
        prediction on each extended feature vector.

        Parameters
        ----------
        oneclass_models : OrderedDict[str, RandomForestClassifier]
            Ordered dictionary of the one-class models, keyed by activity name in order used for
            training
        multiclass_model : RandomForestClassifier
            The multi-class model trained on the original + oc feats feature vectors
        xdata : List[List[float]]
            Input (original) feature vectors to predict on

        Returns
        -------
        List[str]
            Predicted labels from multi-class classifier for each vector in xdata
        """

        oc_xdata = list()

        # Add one-class predictions for each input feature vector:
        for xpoint in xdata:
            oc_xdata.append(MultiOC.add_oc_predictions(xpoint, oneclass_models))

        # Now make the predictions on these new features with multi-class model:
        new_labels = multiclass_model.predict(oc_xdata)

        return new_labels


if __name__ == '__main__':
    """
    Initializations done here because they need to happen whenever we are using this file, 
    including in child processes.
    """

    cf = config.Config(description='AL Activity Learning')
    cf.set_parameters()

    moc = MultiOC(conf=cf)

    # Load location model if needed:
    if cf.locmodel == 1:
        moc.locclf = moc.location.load_location_model()

    # Load translations if needed:
    if cf.translate:
        moc.aclass.read_activity_mappings()
        moc.location.read_location_mappings()

    # Read cached locations from file:
    moc.location.read_locations()

    if cf.mode in [config.MODE_TRAIN_MODEL, config.MODE_TEST_MODEL, config.MODE_LEAVE_ONE_OUT]:
        # We need feature vectors and labels for these modes, so extract the features:
        data_by_file = al.gather_features_by_file(files=cf.files, al=moc)

        if cf.mode == config.MODE_LEAVE_ONE_OUT:
            raise NotImplementedError("Need to implement this soon NOTE: NEED TO USE ALL FILES' ACTIVITIES FOR OC")
        else:
            # We need the total sum of data across all files for train/test, so combine them:
            total_xdata = list()
            total_ydata = list()

            for data in data_by_file.values():
                total_xdata = total_xdata + data[0]
                total_ydata = total_ydata + data[1]

            # Now actually train/test:
            if cf.mode == config.MODE_TRAIN_MODEL:
                moc.train_model(total_xdata, total_ydata)
            elif cf.mode == config.MODE_TEST_MODEL:
                raise NotImplementedError("Need to add this soon")
    else:
        msg = f"Mode {cf.mode} is not supported by this script."
        raise ValueError(msg)
