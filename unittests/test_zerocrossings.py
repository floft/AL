from unittest import TestCase

import numpy

from features import zero_crossings


class ZeroCrossingsTest(TestCase):
    # Testing the zero_crossings function in p.pu
    def test_zero_crossings_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_median = 0.6715
        expected_output = 5

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(expected_output, output)

    def test_zero_crossings_oneentryismedianbutstillcrosses(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_median = 8.004
        expected_output = 4

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(expected_output, output)

    def test_zero_crossings_oneentryismedianbutdoesntcross(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 7.659, 0.143]
        test_median = 8.004
        expected_output = 2

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(expected_output, output)

    def test_zero_crossings_allsamenumber(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        test_median = 5.0
        expected_output = 0

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(expected_output, output)

    def test_zero_crossings_onlyonenumber(self):
        # arrange
        test_data = [4.569]
        test_median = 4.569
        expected_output = 0

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(output, expected_output)

    def test_zero_crossings_emptylist(self):
        # arrange
        test_data = []
        test_median = numpy.nan
        expected_output = 0

        # act
        output = zero_crossings(test_data, test_median)

        # assert
        self.assertEqual(output, expected_output)
