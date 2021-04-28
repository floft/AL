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
