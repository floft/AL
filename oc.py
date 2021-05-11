"""
Code for training and using one-class activity models, which predict a single class at a time. Based
on the code for multi-class model in `al.py`.

Written by Diane J. Cook and Bryan Minor, Washington State University

Copyright (c) 2021. Washington State University (WSU). All rights reserved.
Code and data may not be used or distributed without permission from WSU.
"""
import config
from al import AL, gather_features_by_file


class OC(AL):
    """
    Class to handle training/testing of one-class (OC) models.

    Uses base code from the `AL` class but overridden for one-class models.
    """

    pass


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
        raise NotImplementedError()
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


if __name__ == '__main__':
    main()
