from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from person import calculate_person_features


class CalculatePersonFeaturesTest(TestCase):
    # Testing the calculate_person_features method

    # NOTE: These tests reflect the way the function currently works (returning the cluster index values)
    # We will want to adapt these to our intended ordered ranking (see issue #16 on GitHub)

    def test_calculatepersonfeatures_normaldata_1(self):
        # arrange
        test_infile = None
        test_al = MagicMock()

        test_conf = MagicMock()
        test_conf.num_hour_clusters = 5
        test_conf.numseconds = 5

        test_al.conf = test_conf

        # Set location sensor values:
        test_al.latitude = [45.720049024287654, 45.16015407024762, 45.8461484021051, 45.36231411218953,
                            45.55199912734816, 45.9327322235666]
        test_al.longitude = [-115.33785147412178, -115.8627325962175, -115.7079428458205, -115.74277058776053,
                             -115.95217729730174, -115.99332137643903]
        test_al.altitude = [810.2213245992733, 828.020444363237, 840.4794306806942, 824.2303516812959,
                            819.4818356859511, 829.9659440293694]

        test_person_stats = np.array([
            -1.0,    # (we don't use values before the -4 index, so just use a dummy value here to imitate having preceeding ones)
            46.5,    # meanlat
            -116.0,  # meanlong
            -1.0,    # unused spot
            1.0,     # spanlat
            12.0     # spanlong
        ])

        # Mock up the clusters and set their predict() return values:
        # Note that, since numseconds = 5, we will only use the first 5 location points to predict
        test_clusters = [MagicMock() for _ in range(test_conf.num_hour_clusters)]
        test_clusters[0].predict.return_value = np.array([1, 0, 0, 0, 0])
        test_clusters[1].predict.return_value = np.array([1, 0, 0, 2, 2])
        test_clusters[2].predict.return_value = np.array([2, 1, 1, 2, 2])
        test_clusters[3].predict.return_value = np.array([1, 1, 1, 1, 1])
        test_clusters[4].predict.return_value = np.array([2, 2, 2, 1, 0])

        expected_output = [
            0.07186009116853918,  # distance value

            # most common cluster indices:
            0,
            0,
            2,
            1,
            2
        ]

        # act
        output = calculate_person_features(test_infile, test_al, test_person_stats, test_clusters)

        # assert
        self.assertEqual(output, expected_output)
