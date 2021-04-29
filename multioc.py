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
from datetime import datetime
from typing import Dict, Union, Optional, List, Tuple, OrderedDict

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

    def train_models_for_data(
            self,
            xdata: List[List[float]],
            ydata: List[str],
            activities: List[str]
    ) -> Tuple[OrderedDict[str, object], object]:
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
        Tuple[OrderedDict[str, object], object]
            An ordered dictionary mapping activity names to one-class classifiers (in order that
            features should be applied to existing vectors), and the multi-class classifier.
        """

        pass


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
    else:
        msg = f"Mode {cf.mode} is not supported by this script."
        raise ValueError(msg)
