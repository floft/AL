from unittest import TestCase

import numpy

from features import kurtosis


class KurtosisTest(TestCase):
    # Unit Tests for the Kurtosis Method
    def test_kurtosis_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean = 3.9944999999999995
        test_std = 7.885580225956743
        expected_output = -0.8374466713399569

        # act
        output = kurtosis(test_data, test_mean, test_std)

        # assert
        self.assertEqual(output, expected_output)

    def test_kurtosis_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        test_mean = 5.0
        test_std = 0.0
        expected_output = -3.0

        # act
        output = kurtosis(test_data, test_mean, test_std)

        # assert
        self.assertEqual(output, expected_output)

    def test_kurtosis_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        test_mean = 0.0
        test_std = 0.0
        expected_output = -3.0

        # act
        output = kurtosis(test_data, test_mean, test_std)

        # assert
        self.assertEqual(output, expected_output)

    def test_kurtosis_onlyonenum(self):
        # arrange
        test_data = [1.2]
        test_mean = 1.2
        test_std = 0.0
        expected_output = -3.0

        # act
        output = kurtosis(test_data, test_mean, test_std)

        # assert
        self.assertEqual(output, expected_output)

    def test_kurtosis_emptylist(self):
        # arrange
        test_data = []
        test_mean = numpy.nan
        test_std = numpy.nan

        # act and assert
        with self.assertRaises(ZeroDivisionError):
            kurtosis(test_data, test_mean, test_std)
