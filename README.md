# AL Activity Learning

The AL activity learner learns activity models from activity-labeled mobile
sensor data. The learned models can be used to label new mobile sensor data
with corresponding activity labels.

Author: Dr. Diane J. Cook, School of Electrical Engineering and
Computer Science, Washington State University, email: djcook@wsu.edu.

Support: This material is based on upon work supported by the National Science
Foundation under Grant Nos. 1954372 and 1543656 and by the National Institutes
of Health under Grant Nos. R41EB029774 and R01AG065218. 


# Running AL

After cloning the Git repository, you will also need to initialize the `mobiledata` submodule.
This allows AL to use the [Mobile AL Data Layer](https://github.com/WSU-CASAS/Mobile-AL-Data) for
processing CSV data files. To do so, first go into the main AL directory you cloned and run these
commands:
```
git submodule init
git submodule update
```

AL also requires Pythong packages `geopy`, `numpy`, `requests` and `scikit-learn`.  To install the 
packages for your environment, run:
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
Annotation method to use for labeling new data. Default is 0.

A value of zero (the default) should be used when doing `train`, `test`, or `cv` modes. 

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

Number of data seconds to include in a single window. Default is 30.

Note that the fixed data sample rate set in the code is 10 samples/second. Thus,
a value of `30` for this parameter will result in a window size of 300 samples. (If you wish
to change the sample rate, it can be edited in `config.py`.)

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
`['Errands', 'Exercise', 'Hobby', 'Housework', 'Hygiene', 'Mealtime', 'Other', 'Relax', 'Sleep', 
'Socialize', 'Travel', 'Work']`.

Note that the activities in this list must each have at least one instance in the training data,
so that there is some data to train a model for them. You also should not have any activity labels
in the training data *without* a matching entry in this list. If using activity translations (see 
below), the *translations* of the training labels must match up instead. (If there are no training 
examples for an activity listed in this activity list, you will encounter errors about activities
not being found in the list, at least in some modes.)

```
--translate
```

NOTE: This option is now enabled by default (you do not have to set `--translate` on the command
line to use it.) If you *don't* want to use translations, set `translate = False` in `config.py`.

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
file `data.instances`. The file is in CSV format, with each line of the file showing the values of 
all sensors at a single timestamp.

The first two lines of the file are special header lines. The first line lists all the columns
in the CSV file, starting with the `stamp` (timestamp), followed by the 17 sensors, and ending with
the `user_activity_label`. The second row specifies the data type for each of the columns:
 * `dt` for datetime (timestamp) type
 * `f` for float type
 * `s` for string type

Both header lines should be present in the file, in order to allow the Mobile Data layer to properly
process the CSV file and get data in the proper Python types.

The rest of the lines in the file are the data instances, each with its own timestamp, sensor 
values, and (optionally) a label. Note that all sensor values are also optional. Any missing values
are left blank in when reading in the CSV file, though they are often converted to `0.0` within
the feature extraction code described below.

Multiple files can be specified on the command line. The assumption is that
a separate file of data is used for each person. This distinction will be
important when extracting features for activity learning.

For each data file passed in, AL will start a separate Python `multiprocessing` process to extract
features for that file specifically. Each file's process extracts the features from the data in that
file, and then sends the extracted instances back to the main process via a `multiprocessing` 
`Queue`. Due to limitations with the `Queue`, if an input data file is particularly "large", it may
cause the processing to hang when attempting to send the data back through the `Queue`. In this
case, you will see messages about feature extraction finishing for some files, but then they are not
listed as "done" in the subsequent output, and the main process never finishes. (The point
at which this happens is architecture-dependent. A rough guideline is approximately 500,000 lines).
If any of your input data files are larger than this (or you are finding the extraction hangs),
you may wish to split up the offending file(s) into smaller chunks that will "fit" in the `Queue`.

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
(`latitude` and `longitude`). The person-specific features are stored in a file with the
same root name as the input file and a .person suffix. Based on this stored
information, person-specific features are extracted which include the
normalized distance of the current sequence from the user's mean location and
information related to their frequent "staypoints".  As a note, the person.py script can 
be run separately using any data file, with or without activity labels, to generate 
person-specific features and the associated .person files.

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

# Helpful Tips

## Location Geocoding

Because AL uses reverse geocoding using OpenStreetMap (Nominatim) for features, it needs to
reverse-geocode every unknown location in the input data files. It will cache locations within
a small radius of each other, but other locations must be individually geocoded. The resulting
geocoded locations are then stored in the `locations` file within the same directory as AL.

If the data files are large, AL may take a significant amount of time geocoding the locations,
greatly increasing the feature extraction runtime. In order to avoid this, you may wish to
"pre-geocode" the locations in the data files, before running AL. To do this, right each of the
locations into the file as a separate line in the format:
```
latitude longitude
```

For example:
```
49.21 -117.01
49.25 -117.3
```

Then, run the `gps.py` script on this file as input, which will output the cached geocoded locations
to a `locations` file in the same directory:
```
python gps.py <input_location_tuples_file> 
```

Once the `locations` file has been created, you can then run `al.py` and other scripts as normal,
and they will use the pre-geocoded locations.

## Location File Updates

The final step in the `al.py` script is to update the locations file with any new locations found
while running (see the end of the `main()` function). This can also take a significant amount of
time if there are lots of locations, as AL checks to make sure they do not already exist in the
file. This manifests as AL seeming to finish training/testing models, but then taking a long time to
finish.

In order to reduce this in the case where you have already created the `locations` file (see above),
you may comment out the `gps.update_locations()` call in `main()`.

