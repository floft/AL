from unittest import TestCase

import numpy

from features import mean_crossings


class MeanCrossingsTest(TestCase):
    # Testing the mean_crossings method
    def test_mean_crossings_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean = 3.994499
        expected_output = 4

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_mean_crossings_oneentryismeanbutstillcrosses(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean = 8.004
        expected_output = 4

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_mean_crossings_oneentryismeanbutdoesntcross(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 7.659, 0.143]
        test_mean = 8.004
        expected_output = 2

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_mean_crossings_allsamenumber(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        test_mean = 5.0
        expected_output = 0

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_mean_crossings_onlyonenumber(self):
        # arrange
        test_data = [4.569]
        test_mean = 4.569
        expected_output = 0

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)

    def test_mean_crossings_emptylist(self):
        # arrange
        test_data = []
        test_mean = numpy.nan
        expected_output = 0

        # act
        output = mean_crossings(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)
