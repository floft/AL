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
from sklearn.metrics import confusion_matrix, classification_report

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

    def save_models(
            self,
            oneclass_models: OrderedDict[str, RandomForestClassifier],
            multiclass_model: RandomForestClassifier
    ):
        """
        Save the given one-class models and multi-class model to disk as pickle file.

        The models will actually be saved as a pickled dictionary with keys:
         - oneclass_models: OrderedDict of activity->model pairs for one-class models (in their
           order)
         - multiclass_model: The actual multi-class model

        Parameters
        ----------
        oneclass_models : OrderedDict[str, RandomForestClassifier]
            Ordered dictionary of the one-class models, keyed by activity name in order used for
            training
        multiclass_model : RandomForestClassifier
            The multi-class model trained on the original + oc feats feature vectors
        """

        print("Saving models...", end='')

        models_dict = {
            'oneclass_models': oneclass_models,
            'multiclass_model': multiclass_model
        }

        outstr = self.conf.modelpath + 'multioc_model.pkl'
        joblib.dump(models_dict, outstr)

        print("done")

    def load_models(self) \
            -> Tuple[OrderedDict[str, RandomForestClassifier], RandomForestClassifier]:
        """
        Load the one-class and multi-class classifiers, returning an OrderedDict of the one-class
        classifiers followed by the multi-class classifier.

        Expects the models to be stored in a dictionary (see `save_models()` method for format).
        """

        models_filename = self.conf.modelpath + 'multioc_model.pkl'

        with open(models_filename, 'rb') as f:
            models_dict = joblib.load(f)  # type: Dict[str, Union[OrderedDict[str, RandomForestClassifier], RandomForestClassifier]]

        oneclass_models = models_dict['oneclass_models']
        multiclass_model = models_dict['multiclass_model']

        return oneclass_models, multiclass_model

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
        self.save_models(oneclass_models, multiclass_model)

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

        new_xdata, new_ydata = self.translate_and_filter_labels(oc_xdata, ydata)

        print("done")

        # Now train the multi-class classifier on the new data:
        # (self.clf is the AL multi-class classifier configured instance)
        print("Training multi-class classifier...", end='')

        self.clf.fit(new_xdata, new_ydata)

        print("done")

        return oc_classifiers, self.clf

    def translate_and_filter_labels(self, xdata: List[List[float]], ydata: List[str]) \
            -> Tuple[List[List[float]], List[str]]:
        """
        Process the provided feature vectors and labels as follows:
         - Translate labels if needed
         - Only include a feature vector/label pair in output if the (translated) label is not
           'Ignore'

        Parameters
        ----------
        xdata : List[List[float]]
            Input (original) feature vectors
        ydata : List[str]
            Input (original) activity labels corresponding to the feature vectors

        Returns
        -------
        Tuple[List[List[float]], List[str]]
            The new (translated and filtered) (xdata, ydata) pair
        """

        new_xdata = list()
        new_ydata = list()

        for i, orig_label in enumerate(ydata):
            new_label = orig_label

            if self.conf.translate:
                new_label = self.aclass.map_activity_name(orig_label)

            # Only add instances that don't have 'Ignore' or 'None' for labels:
            if new_label != 'Ignore' and new_label != 'None':
                new_xdata.append(xdata[i])
                new_ydata.append(new_label)

        return new_xdata, new_ydata

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

    def test_model(self, xdata: list, ydata: list):
        """
        Test saved models on the provided features (`xdata`) and un-translated labels (`ydata`).

        Override so that we load the proper models, and also do the translation of labels (if
        needed) and removing 'Ignore' and 'None' labeled vectors from contention.

        Parameters
        ----------
        xdata : list
            Input (original) feature vectors to predict on
        ydata : list
            Original (un-translated, and possibly including 'Ignore' activities) ground-truth labels
        """

        # Load the models:
        oneclass_models, multiclass_model = self.load_models()

        # Now run the test:
        self.test_models(oneclass_models, multiclass_model, xdata, ydata)

    def test_models(
            self,
            oneclass_models: OrderedDict[str, RandomForestClassifier],
            multiclass_model: RandomForestClassifier,
            xdata: List[List[float]],
            ydata: List[str]
    ):
        """
        Test saved models on the provided features (`xdata`) and un-translated labels (`ydata`)
        using the given one-class and multi-class classifiers.

        Includes doing the translation of labels (if needed) and removing 'Ignore' and 'None'
        labeled vectors from contention.

        Parameters
        ----------
        oneclass_models : OrderedDict[str, RandomForestClassifier]
            Ordered dictionary of the one-class models, keyed by activity name in order used for
            training
        multiclass_model : RandomForestClassifier
            The multi-class model trained on the original + oc feats feature vectors
        xdata : List[List[float]]
            Input (original) feature vectors to predict on
        ydata : List[str]
            Original (un-translated, and possibly including 'Ignore' activities) ground-truth labels
        """

        # First translate the ydata labels and only include instances that aren't labeled 'Ignore':
        new_xdata, new_ydata = self.translate_and_filter_labels(xdata, ydata)

        # Now make predictions on the new xdata:
        new_labels = self.test_models_for_data(oneclass_models, multiclass_model, new_xdata)

        # Get list of activities in the new ydata:
        activity_list = sorted(set(new_ydata))

        # Now output results:
        print('Activities:', ' '.join(activity_list))
        print('   (i,j) means true label is i, predicted as j')

        numright = total = 0

        matrix = confusion_matrix(new_ydata, new_labels, labels=activity_list)

        # Calculate the accuracy:
        for j in range(len(new_ydata)):
            if new_labels[j] == new_ydata[j]:
                numright += 1
            total += 1

        print('test accuracy', float(numright) / float(total), '\n', matrix)

        print(classification_report(new_ydata, new_labels))

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

    def leave_one_out(self, file_data: Dict[str, Tuple[List, List]]):
        """
        Perform leave-one-out testing on the data, by cycling through the list of files. For each
        file, train the models on all the other files' data, then test them on that file's data.

        Parameters
        ----------
        file_data : Dict[str, Tuple[List, List]]
            (xdata, ydata) tuples for each file
        """

        if len(file_data) < 2:
            error_msg = "Need to have at least 2 files for leave-one-out testing"
            raise ValueError(error_msg)

        print("Starting leave-one-out")

        # Get the overall list of activities across all files:
        all_files_ydata = list()

        for (_, ydata) in file_data.values():
            all_files_ydata.extend(ydata)

        oc_activities = self.get_oneclass_labels(all_files_ydata)

        # Now do the actual leave-one-out testing:
        for test_file in file_data.keys():
            print(f"Leaving {test_file} out")

            # Get all other files except this one:
            train_files = [f for f in file_data.keys() if f != test_file]

            # Create the training data from other files:
            train_xdata = list()
            train_ydata = list()

            for train_file in train_files:
                train_xdata += file_data[train_file][0]
                train_ydata += file_data[train_file][1]

            # Train the models:
            oc_models, mc_model = \
                self.train_models_for_data(train_xdata, train_ydata, oc_activities)

            # Now test those models:
            print(f"Test results for left-out file {test_file}")

            test_xdata = file_data[test_file][0]
            test_ydata = file_data[test_file][1]

            self.test_models(oc_models, mc_model, test_xdata, test_ydata)


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
            moc.leave_one_out(data_by_file)
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
                moc.test_model(total_xdata, total_ydata)
    else:
        msg = f"Mode {cf.mode} is not supported by this script."
        raise ValueError(msg)
