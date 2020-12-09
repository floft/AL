# activity.py
# Handle activity label processing.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


class Activity:

    def __init__(self):
        """ Constructor
        """
        self.amappings = dict()
        self.amappings['Other'] = 'Other'
        self.amappings2 = dict()
        self.amappings2['Other'] = 'Other'

    def map_activity_name(self, name):
        """ Return the (more general) activity label that is associated with a
        specific activity name, using a stored list of activity mappings.
        """
        newname = self.amappings.get(name)
        if newname is None:
            return 'Other'
        else:
            return newname

    def map_activity_name2(self, name):
        """ Return the (more general) activity label that is associated with a
        specific activity name, using a stored list of activity mappings.
        """
        newname = self.amappings2.get(name)
        if newname is None:
            return 'Other'
        else:
            return newname

    def read_activity_mappings(self):
        """ Generate a translate list for activity names.
        This function assumes that file act.translate exists in the same
        directory as the code. File act.translate contains an arbitrary number
        of lines, each with syntax "specificType mappedType". This function maps
        activities of specificType to the corresponding, more general, mappedType.
        """
        with open('act.translate', "r") as file:
            for line in file:
                x = str(str(line).strip()).split(' ', 3)
                self.amappings[x[0]] = x[1]

    def read_activity_mappings_both(self):
        """ Generate a translate list for activity names.
        This function assumes that file act.translate exists in the same
        directory as the code. File act.translate contains an arbitrary number
        of lines, each with syntax "specificType mappedType". This function maps
        activities of specificType to two corresponding, more general, mappedTypes.
        Two types can be used effectively when one is a more general and the other
        is a more specific type, or when the activity can be viewed as an instance
        of two distinct categories.
        """
        with open('oca.translate', "r") as file:
            for line in file:
                x = str(str(line).strip()).split(' ', 3)
                self.amappings[x[0]] = x[1]
                self.amappings2[x[0]] = x[2]
