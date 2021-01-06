from unittest import TestCase

from features import autocorrelation


class AutoCorrelationTest(TestCase):
    # Unit tests for the autocorrelation method
    def test_autocorrelation_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean = 3.9944999999999995
        expected_output = -0.5801193286061433

        # act
        output = autocorrelation(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)

    def test_autocorrelation_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        test_mean = 5.0
        expected_output = 0.0

        # act
        output = autocorrelation(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)

    def test_autocorrelation_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        test_mean = 0.0
        expected_output = 0.0

        # act
        output = autocorrelation(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)

    def test_autocorrelation_emptylist(self):
        # arrange
        test_data = []
        test_mean = None

        # act and assert
        with self.assertRaises(ZeroDivisionError):
            autocorrelation(test_data, test_mean)

    def test_autocorrelation_onedatapoint(self):
        # arrange
        test_data = [3.2]
        test_mean = 3.2

        # act and assert
        with self.assertRaises(ZeroDivisionError):
            autocorrelation(test_data, test_mean)

    def test_autocorrelation_twodatapoints(self):
        # arrange
        test_data = [1.2, 0.05]
        test_mean = 0.625
        expected_output = -2.0

        # act
        output = autocorrelation(test_data, test_mean)

        # assert
        self.assertEqual(output, expected_output)
