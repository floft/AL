from unittest import TestCase

import numpy

from features import skewness


class SkewnessTest(TestCase):
    # Unit testing for Skewness Method
    def test_skewness_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean = 3.9944999999999995
        expected_output = 0.6676601713176257

        # act
        output = skewness(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_skewness_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        test_mean = 5.0
        expected_output = 0.0

        # act
        output = skewness(test_data, test_mean)

        # assert
        self.assertEqual(expected_output, output)

    def test_skewness_onlyonenum(self):
        # arrange
        test_data = [4.569]
        test_mean = 4.569
        expected_output = 0.0

        # act
        output = skewness(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)

    def test_skewness_emptylist(self):
        # arrange
        test_data = []
        test_mean = numpy.isnan

        # act and assert
        with self.assertRaises(ZeroDivisionError):
            skewness(test_data, test_mean)

    def test_skewness_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        test_mean = 0.0
        expected_output = 0.0

        # act
        output = skewness(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)