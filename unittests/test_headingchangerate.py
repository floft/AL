from unittest import TestCase

from features import heading_change_rate


class HeadingChangeRateTest(TestCase):
    # unit tests for the heading_change_rate method

    def test_heading_change_rate_regulardata(self):
        # arrange
        test_data = [1, 28.98, 340.01, 254, 58.9, 168.4, 23, 15.0, 101.111, 359.0]
        test_distance = 0.0005
        expected_output = 18000.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_allzeros(self):
        # arrange
        test_data = [0.000, 0.0, 0, 0.0, 0000.0, 00.0, 0.0, 0.0, 0, 0]
        test_distance = 0.0005
        expected_output = 0.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5, 5.0, 5, 5.00]
        test_distance = 0.0005
        expected_output = 0.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_regulardatawithinvalidentries(self):
        # arrange
        test_data = [1, -1.0, 340.01, 254, -1.0, 168.4, 23, 15.0, 101.111, -1.0]
        test_distance = 0.0005
        expected_output = 12000.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_twodatapoints(self):
        # arrange
        test_data = [168.53, 28.98]
        test_distance = 0.0005
        expected_output = 2000.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_onedatapoint(self):
        # arrange
        test_data = [168.53]
        test_distance = 0.0005
        expected_output = 0.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_emptylist(self):
        # arrange
        test_data = []
        test_distance = 0.0005

        # act and assert
        with self.assertRaises(IndexError):
            heading_change_rate(test_data, test_distance)

    def test_heading_change_rate_distanceiszero(self):
        # arrange
        test_data = [1, 28.98, 340.01, 254, 58.9, 168.4, 23, 15.0, 101.111, 359.0]
        test_distance = 0.0
        expected_output = 0.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_heading_change_rate_distanceislarge(self):
        # arrange
        test_data = [1, 28.98, 340.01, 254, 58.9, 168.4, 23, 15.0, 101.111, 359.0]
        test_distance = 5129.14
        expected_output = 0.0017546801218137932

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)


    def test_heading_change_rate_distanceisnegative(self):
        # arrange
        test_data = [1, 28.98, 340.01, 254, 58.9, 168.4, 23, 15.0, 101.111, 359.0]
        test_distance = -0.0005
        expected_output = -18000.0

        # act
        output = heading_change_rate(test_data, test_distance)

        # assert
        self.assertEqual(output, expected_output)