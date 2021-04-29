# config.py
# Global variables

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.

import argparse
import copy

MODE_CROSS_VALIDATION = 'cv'
MODE_LEAVE_ONE_OUT = 'loo'
MODE_TRAIN_MODEL = 'train'
MODE_TEST_MODEL = 'test'
MODE_ANNOTATE_DATA = 'ann'
MODES = list([MODE_CROSS_VALIDATION,
              MODE_LEAVE_ONE_OUT,
              MODE_TRAIN_MODEL,
              MODE_TEST_MODEL,
              MODE_ANNOTATE_DATA])
MODE_AL_CROSS_VALIDATION = 'al_cross_validation'
MODE_AL_TRAIN_MODEL = 'al_train_model'
MODE_AL_TEST_MODEL = 'al_test_model'
MODE_AL_ANNOTATE_DATA = 'al_annotate_data'
MODE_LOC_CROSS_VALIDATION = 'loc_cross_validation'
MODE_LOC_TRAIN_MODEL = 'loc_train_model'
MODE_OC_CROSS_VALIDATION = 'oc_cross_validation'
MODE_OC_TRAIN_MODEL = 'oc_train_model'
MODE_OC_TEST_MODEL = 'oc_test_model'
MODE_OC_ANNOTATE_DATA = 'oc_annotate_data'

STAMP_CSV_FIELD = 'stamp'  # name of the timestamp field in CSV files
ACTIVITY_LABEL_CSV_FIELD = 'user_activity_label'  # name of the user activity label field in CSV


class Config:

    def __init__(self, description: str):
        """ Constructor
        """
        self.description = description

        # Define the args we use for the various programs.

        # The core activity we will perform.
        self.mode = MODE_CROSS_VALIDATION

        # Number of jobs to run section.
        # Number of jobs to tell the AL models to use when available.
        self.al_n_jobs = 20
        # Number of jobs to tell the OC models to use when available.
        self.oc_n_jobs = 20
        # Number of jobs to tell the Location models to use when available.
        self.loc_n_jobs = 1

        self.numsensors = 16  # Number of sensors generating readings each sample
        self.samplerate = 1  # Number of sensor readings per second
        self.numseconds = 5  # Number of data seconds to include in a single window

        # Number of reading sets in one window
        self.samplesize = self.samplerate * self.numseconds

        # Number of sensor readings in one window
        self.windowsize = self.samplesize * self.numsensors
        self.sfeatures = 1  # Use spatial features
        self.translate = False  # Map activities to smaller set of activity names
        self.gpsfeatures = 1  # Use gps features
        self.fftfeatures = 1  # Use FFT features
        self.personfeatures = 1  # Use person-specific features
        self.locmodel = 0  # Use learned model to generate a location type
        self.filter_data = True  # Apply signal processing filters to sensor data
        self.local = 1  # Use the local GPS values
        self.gen_gps_abs_stat_features = False  # Generate "absolute" statistical features for GPS sensors
        self.cv = 3  # cross validation number of folds
        self.annotate = 0  # label new data from learned model
        self.extension = '.instances'  # filename extension for training data
        self.datapath = './data/'  # directory containing files of sensor data
        self.modelpath = './models/'  # directory containing trained models
        self.clusterpath = './clusters/'  # directory of personal location clusters
        self.num_hour_clusters = 5  # number of clusters stored for each person

        # Defaults for the CSV field headers of interest
        self.stamp_field_name = STAMP_CSV_FIELD
        self.label_field_name = ACTIVITY_LABEL_CSV_FIELD

        # default list of activity classes for overall activity
        self.default_activity_list = ['Chores', 'Eat', 'Entertainment', 'Errands',
                                      'Exercise', 'Hobby', 'Hygiene', 'Relax', 'School',
                                      'Sleep', 'Travel', 'Work']
        # list of activity classes for overall activity
        self.activity_list = list()

        # default list of one-class activity classes
        self.default_activities = ['Airplane', 'Art', 'Bathe', 'Biking', 'Bus', 'Car',
                                   'Chores', 'Church', 'Computer', 'Cook', 'Dress', 'Drink',
                                   'Eat', 'Entertainment', 'Errands', 'Exercise', 'Groom',
                                   'Hobby', 'Hygiene', 'Lunch', 'Movie', 'Music', 'Relax',
                                   'Restaurant', 'School', 'Service', 'Shop', 'Sleep',
                                   'Socialize', 'Sport', 'Travel', 'Walk', 'Work']
        self.activities = list()
        self.numactivities = len(self.default_activities) + 1
        self.oneclass = False
        self.multioc_ground_truth_train = False  # should multioc use ground-truth one-class feats
        self.files = list()
        return

    def set_parameters(self):
        """ Set parameters according to command-line args list.
        """
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('--mode',
                            dest='mode',
                            type=str,
                            choices=MODES,
                            default=self.mode,
                            help=('Define the core mode that we will run in, default={}.'
                                  .format(self.mode)))
        parser.add_argument('--cv',
                            dest='cv',
                            type=int,
                            default=self.cv,
                            help=('The number of cross validations to perform, default={}.'
                                  .format(self.cv)))
        parser.add_argument('--annotate',
                            dest='annotate',
                            type=int,
                            choices=[1, 2],
                            default=self.annotate,
                            help=('Label new data from a learned model if 1 or 2, default={}.'
                                  .format(self.annotate)))
        parser.add_argument('--numseconds',
                            dest='numseconds',
                            type=int,
                            default=self.numseconds,
                            help=('Number of data seconds to include in a single window, '
                                  'default={}.'.format(self.numseconds)))
        parser.add_argument('--extension',
                            dest='extension',
                            type=str,
                            default=self.extension,
                            help=('Filename extension for training data, default={}'
                                  .format(self.extension)))
        parser.add_argument('--datapath',
                            dest='datapath',
                            type=str,
                            default=self.datapath,
                            help=('Directory containing files of sensor data, default={}'
                                  .format(self.datapath)))
        parser.add_argument('--modelpath',
                            dest='modelpath',
                            type=str,
                            default=self.modelpath,
                            help=('Directory containing trained models, default={}'
                                  .format(self.modelpath)))
        parser.add_argument('--clusterpath',
                            dest='clusterpath',
                            type=str,
                            default=self.clusterpath,
                            help=('Directory of personal location clusters, default={}'
                                  .format(self.clusterpath)))
        parser.add_argument('--activities',
                            dest='activities',
                            type=str,
                            default=','.join(self.default_activities),
                            help=('Comma separated list of one-class activity classes, default='
                                  '{}'.format(','.join(self.default_activities))))
        parser.add_argument('--activity-list',
                            dest='activity_list',
                            type=str,
                            default=','.join(self.default_activity_list),
                            help=('Comma separated list of activity classes for overall activity, '
                                  'default={}'.format(','.join(self.default_activity_list))))
        parser.add_argument('--translate',
                            dest='translate',
                            default=self.translate,
                            action='store_true',
                            help=('Map activities to smaller set of activity names, default={}'
                                  .format(self.translate)))
        parser.add_argument('--oneclass',
                            dest='oneclass',
                            default=self.oneclass,
                            action='store_true',
                            help=('Learn one-class classifier for each activity in --activities, '
                                  'default={}'.format(self.oneclass)))
        parser.add_argument('--multioc-ground-truth',
                            dest='multioc_ground_truth',
                            default=self.multioc_ground_truth_train,
                            action='store_true',
                            help=('Use ground-truth labels for one-class features for multioc'
                                  'default=%(default)s'))
        parser.add_argument('files',
                            metavar='FILE',
                            type=str,
                            nargs='+',
                            help='Data files for AL to process.')
        parser.add_argument('--aljobs',
                            dest='aljobs',
                            type=int,
                            default=self.al_n_jobs,
                            help=('The number of jobs to tell the AL models to use when available, '
                                  'default={}.'.format(self.al_n_jobs)))
        parser.add_argument('--ocjobs',
                            dest='ocjobs',
                            type=int,
                            default=self.oc_n_jobs,
                            help=('The number of jobs to tell each OneClass model to use when '
                                  'available, default={}.'.format(self.oc_n_jobs)))
        parser.add_argument('--locjobs',
                            dest='locjobs',
                            type=int,
                            default=self.loc_n_jobs,
                            help=('The number of jobs to tell the Location model to use when '
                                  'available, default={}.'.format(self.loc_n_jobs)))
        args = parser.parse_args()

        self.mode = args.mode
        self.cv = int(args.cv)
        self.annotate = int(args.annotate)
        self.numseconds = int(args.numseconds)
        self.extension = args.extension
        self.datapath = args.datapath
        self.modelpath = args.modelpath
        self.clusterpath = args.clusterpath
        self.activities = str(args.activities).split(',')
        self.numactivities = len(self.activities) + 1
        self.activity_list = str(args.activity_list).split(',')
        self.translate = bool(args.translate)
        self.oneclass = bool(args.oneclass)
        self.multioc_ground_truth_train = bool(args.multioc_ground_truth)
        self.al_n_jobs = int(args.aljobs)
        self.oc_n_jobs = int(args.ocjobs)
        self.loc_n_jobs = int(args.locjobs)
        self.files = copy.deepcopy(args.files)

        return args.files
