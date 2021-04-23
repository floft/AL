from unittest import TestCase

import person
from config import Config


class PersonReadLocationsTest(TestCase):
    """
    Test the person.read_locations() function.
    """

    def test_readlocations_normaldata(self):
        # Arrange:
        test_config = Config(description='test')
        test_config.datapath = './test_data/'  # point to the test data directory
        test_config.extension = '.csv'  # use the CSV file

        test_base_filename = 'multi_location_test'

        # Note that second to last value in the file has a bad latitude, so don't include it
        # Also note that invalid longitude value (-1000.0) is allowed since the lat is still valid
        # on that point
        expected_latitude_list = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
                                  57.1, 57.1, 57.1, 57.1, -2.2, 40.0, 40.0, 11.0, 11.0, 11.0, 11.0,
                                  40.0, 40.0, 40.0, 40.0]
        expected_longitude_list = [-117.0, -117.0, -117.0, -117.0, -117.0, -117.0, -117.0,
                                   -117.0, -117.0, -117.0, -131.2, -131.2, -131.2, -131.2, -3.7,
                                   -117.0, -1000.0, -11.0, -11.0, -11.0, -11.0, -117.0, -117.0,
                                   -117.0, -117.0]
        expected_altitude_list = [748.360107421875, 748.360107421875,
                                  748.360107421875, 748.360107421875, 748.360107421875,
                                  748.360107421875, 748.360107421875, 3.0, 3.0,
                                  748.360107421875, 748.360107421875, 748.360107421875,
                                  748.360107421875, 748.360107421875, 748.360107421875,
                                  748.360107421875, 748.360107421875, 748.360107421875,
                                  -1.3, 748.360107421875, 748.360107421875, 748.360107421875,
                                  748.360107421875, 748.360107421875, 748.360107421875]
        expected_hour_list = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                              15, 15, 18, 18, 18, 18, 18, 18, 12]

        # Act:
        actual_latitude_list, actual_longitude_list, actual_altitude_list, actual_hour_list \
            = person.read_locations(test_base_filename, test_config)

        # Assert:
        self.assertEqual(actual_latitude_list, expected_latitude_list)
        self.assertEqual(actual_longitude_list, expected_longitude_list)
        self.assertEqual(actual_altitude_list, expected_altitude_list)
        self.assertEqual(actual_hour_list, expected_hour_list)
