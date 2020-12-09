# AL Activity Learning

The AL activity learner learns activity models from activity-labeled mobile
sensor data. The learned models can be used to label new mobile sensor data
with corresponding activity labels.

Author: Dr. Diane J. Cook, School of Electrical Engineering and
Computer Science, Washington State University, email: djcook@wsu.edu.

Support: This material is based on upon work supported by the National Science
Foundation under Grant Nos. 1954372 and 1543656 and by the National Institutes
of Health under Grant No. R41EB029774.


# Running AL

AL requires packages `geopy`, `numpy`, `requests` and `scikit-learn`.  To install the packages
for your environment, run:
```
pip install geopy numpy requests scikit-learn
```

AL is run using the following command-line format (requires Python 3):

```
python al.py [options] <inputfiles>
```

The options, input file format, and output are described below.


# Options

The following AL options are currently available. If not provided, each option will be set 
to its default value.

You can also view information on each argument by using
```
python al.py --help
```

```
--mode <mode>
```
Valid modes are `cv`, `train`, `test`, and `ann`.  The default is `cv`.

`cv` will run a cross validation using k-folds, where `k` is the value set by `--cv`.

`train` will train a model on the provided data and save the model to disk.

`test` will load a saved model from disk and test it on the provided data.

`ann` will load a saved model from disk and annotate the provided data files. 

```
--cv <n>
```
Cross validation specification for training activity models. Default is 3.
If `cv` > 0, then activity models are trained and tested using k-fold cross
validation, where `k` is the command-line argument specified for `cv`. Performance
is reported based on accuracy and a corresponding confusion matrix is output.

```
--annotate <n>
```
Annotation method to use for labeling new data. Default is 1.

With `--annotate 1`, the output is created in the same format as the input
(one line per sensor value) with the last value being replaced by the
overall activity label. The output is stored in a file with the same name
as the input file and suffix `.ann`.

With `--annotate 2`, a csv file is created, with one line for each set of 
sensor readings with the same time stamp. The format of each line is
`day,time,<sensor values>,<activity label(s)>`. The output is stored in a file
with the same name as the input file and suffix `.anncombine`.

```
--numseconds <n>
```

Number of data seconds to include in a single window. Default is 5.

Note that the fixed data sample rate set in the code is 1 sample/second. Thus,
a value of `5` for this parameter will result in a window size of 5 samples.

```
--extension <extension_string>
```

Filename extension appended to training data filenames. Default value is `'.instances'`.

This extension will be appended to all filenames passed in `<inputfiles>` above. For 
example, given an input file name `'data'` with the default extension, the program will
look for `'data.instances'` for training data in the directory defined by `--datapath`.

```
--datapath <path_string>
```
Specification of the location for the data files.
The default value is `./data/`.

```
--modelpath <path_string>
```
Specification of the location where models are stored.
The default value is `./models/`.

```
--clusterpath <path_string>
```
Specification of the location where person-specific cluster models are stored.
The default value is `./clusters/`.

```
--activities <list>
```
Specify list of activities to learn with one-class classifiers. There needs to
be at least one occurrence of each activity in the training data. The default
value is the list `['Airplane', 'Art', 'Bathe', 'Biking', 'Bus', 'Car', 'Chores', 'Church',
'Computer', 'Cook', 'Dress', 'Drink', 'Eat', 'Entertainment', 'Errands', 'Exercise',
'Groom', 'Hobby', 'Hygiene', 'Lunch', 'Movie', 'Music', 'Relax', 'Restaurant', 'School',
'Service', 'Shop', 'Sleep', 'Socialize', 'Sport', 'Travel', 'Walk', 'Work']`.

```
--activity_list <list>
```
Specify list of overall (primary) activities to learn with a single
multi-class classifier. There should be at least one occurrence of each
activity in the training data. The default value is the list
`['Chores', 'Eat', 'Entertainment', 'Errands', 'Exercise', 'Hobby', 'Hygiene', 'Relax',
'School', 'Sleep', 'Travel', 'Work']`.

```
--translate
```
If this option is provided, then activity names and location names will be
mapped from their original values to a different (typically smaller) set of
values. This option is generally used to compress the number of different
activity classes and location types that are considered.

When this option is used, AL will look in file `act.translate` for a
set of activity mappings, one per line. Each line contains the original
activity label and the new mapped label, separated by a space. For example,
if the file contains the entries
```
Breakfast Eat
Lunch Eat
Dinner Eat
Work_At_Home Work
Work_At_Office Work
```
then activities `Breakfast`, `Lunch`, and `Dinner` will be mapped to a single activity
class `Eat`, while `Work_At_Home` and `Work_At_Office` will be mapped to the activity
class `Work`.

When this option is used, AL will also look in file `loc.translate` for a
set of location mappings, one per line. Each line contains the original
activity label and the new mapped label, separated by a space. Examples are
```
office work
park service
parking road
```
When the open street map returns type `office` this will be mapped to `work`,
`park` will be mapped to `service`, and `parking` will be mapped to `road`.

If this option is used together with the oneclass option, then AL will also
look in file `oca.translate` for a set of activity mappings, one per line.
Each line contains the original activity label and two mapped labels, all
separated by a space. For example,
```
Wash_Face Groom Hygiene
Wash_Hands Groom Hygiene
Washing_Dishes Cook Chores
Work_Out Exercise Exercise
```
maps the original activity labels `Wash_Face` and `Wash_Hands` to both `Groom` and
`Hygiene`, while activity label `Washing_Dishes` is mapped to both `Cook` and `Chores`.
Each of the new activity categories is learned by a one-class classifier, so
multiple activities can be used to express the current task. If only one
activity label is appropriate it can be used twice. In this example, `Workout`
is only mapped to `Exercise`.

```
--oneclass
```
This option specifies that multiple types of activity models will be learned.
One model is a multi-class classifier that maps a feature vector describing
a sequence of sensor readings onto a single activity label from a list
(the list specified by `activity_list`). In addition, multiple binary classifiers
are trained to recognize whether the sequence is an instance of a particular
activity type (the list is specified by activities). This is because due to
semantic ambiguities, the current situation may be represented by more than one
activity class.

When the oneclass option is used with training mode (`--mode`=`train`), each of the
models is trained using the input data. The multi-class classifier is stored
as `model.pkl` in the models directory, while the binary classifiers are stored
as their activity name (e.g., `Work.pkl`) in the models directory.

When the oneclass option is used in annotation mode (`--mode`=`ann`), a csv file
is generated. The csv file contains one line per time stamp. Each line includes
the day, time, each sensor reading, a 0 or 1 value for each binary classifier
prediction, and a value for the multi-class classifier.


# Input File(s)

The input file(s) contains time-stamped sensor readings. An example is in the
file `data.instances`. Each line of the input file contains a reading for
a single sensor. The current version of AL assumes that there are 16 sensors.
For any timepoint there will be 16 lines, each which has the same time and
date but differ in the sensor name and value. An example is shown below.
```
2017-03-03 11:51:55.062000 Yaw Yaw 0.245832 0
2017-03-03 11:51:55.062000 Pitch Pitch 0.011857 0
2017-03-03 11:51:55.062000 Roll Roll -0.001035 0
2017-03-03 11:51:55.062000 RotationRateX RotationRateX -3.7e-05 0
2017-03-03 11:51:55.062000 RotationRateY RotationRateY 0.001487 0
2017-03-03 11:51:55.062000 RotationRateZ RotationRateZ 0.001275 0
2017-03-03 11:51:55.062000 UserAccelerationX UserAccelerationX -0.002322 0
2017-03-03 11:51:55.062000 UserAccelerationY UserAccelerationY 0.000656 0
2017-03-03 11:51:55.062000 UserAccelerationZ UserAccelerationZ 0.006353 0
2017-03-03 11:51:55.062000 Latitude Latitude 33.74305 0
2017-03-03 11:51:55.062000 Longitude Longitude -107.183205 0
2017-03-03 11:51:55.062000 Altitude Altitude 787.196045 0
2017-03-03 11:51:55.062000 Course Course -1.0 0
2017-03-03 11:51:55.062000 Speed Speed 0.0 0
2017-03-03 11:51:55.062000 HorizontalAccuracy HorizontalAccuracy 10.0 0
2017-03-03 11:51:55.062000 VerticalAccuracy VerticalAccuracy 3.0 Work
```
The general format for the data contains 6 fields per line. The fields are:

* date: `yyyy-mm-dd`
* time: `hh:mm:ss.ms`
* sensor name
* sensor name (in the current version the same name appears twice)
* sensor reading
* label: this is either 0 or a string indicating the activity label (this field
         is required on all lines, but the actual activity label string is only
         required on the last line for each timestamp)

Multiple files can be specified on the command line. The assumption is that
a separate file of data is used for each person. This distinction will be
important when extracting features for activity learning.

NOTE: Each input file name passed as `<inputfiles>` when calling `al.py` will have 
the string from `--extension` appended to the end of it when processing.


# Features

AL extracts a vector of features for each non-overlapping sensor sequence of
length `windowsize` (`windowsize` indicates the number of distinct time stamps that
are included in the sequence). A random forest classifier maps the feature
vector to the activity label(s).

The first set of features AL extracts are person-specific features related to
location. Specific <`latitude`, `longitude`, `altitude`> values do not generalize
well to multiple people, so more abstract features are used. Before activity
models are learned, each input file is processed separately to construct
person-specific information. The information includes the person's mean
location (`latitude` and `longitude`) and the span of their visited locations
(`latitude` and `longitude`). To identify frequent locations, k-means clustering
is applied to determine the overall top location clusters and top location
clusters by time of day. The learned cluster models are stored in the clusters
directory and the person-specific features are stored in a file with the
same root name as the input file and a .person suffix. Based on this stored
information, person-specific features are extracted which include the
normalized distance of the current sequence from the user's mean location and
whether the last location in the current sequence belongs to any of the
learned clusters.  As a note, the person.py script can be run separately using any
data file, with or without activity labels, to generate person-specific features
and the associated cluster and .person files.

Additional statistical features are applied to each non-location sensor
including max, min, sum, mean, median, mean/median absolute value, standard
deviation, mean/median absolute deviation, moments, fft features (ceps, entropy,
energy, msignal, and vsignal), coefficient of variation, skewness, kurtosis,
signal energy, log signal energy, power, autocorrelation, the absolute
differences between successive values in the sequence, and time between peaks.
For sensors with multiple axes (e.g., acceleration, rotation), correlations
between the axes are added to the feature vector. Before extracting these
features, a lowpass filter is applied to remove signal noise.

For the location values, features further include the heading change rate
(number of orientation changes within the sequence), stop rate (number of
movement stops/starts within the sequence), and overall trajectory in
the sequence. Additionally, reverse geocoding using OpenStreetMap
and the `loc.translate` file is used to add the location type to the feature list.

