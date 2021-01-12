from unittest import TestCase

import numpy

from features import mean_absolute_deviation


class MeanAbsoluteDeviationTest(TestCase):
    # Testing the mean absolute deviation function from p.py
    def test_mean_absolute_deviation_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        expected_output = 6.894875

        # act
        output = mean_absolute_deviation(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_mean_absolute_deviation_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        expected_output = 0.0

        # act
        output = mean_absolute_deviation(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_mean_absolute_deviation_allsamenumber(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        expected_output = 0.0

        # act
        output = mean_absolute_deviation(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_mean_absolute_deviation_onlyonenumber(self):
        # arrange
        test_data = [4.569]
        expected_output = 0.0

        # act
        output = mean_absolute_deviation(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_mean_absolute_deviation_emptylist(self):
        # assert
        test_data = []

        # act
        output = mean_absolute_deviation(test_data)

        # assert
        self.assertTrue(numpy.isnan(output))
