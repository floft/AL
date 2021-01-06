from unittest import TestCase

from features import correlation


class CorrelationTest(TestCase):
    # Unit tests for the correlation method
    def test_correlation_regulardata(self):
        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_y = [7.3, -1.3,  4.1, 0.12, 0.05,   7.1,   2.12, 6.713]
        test_mean_x = 3.9944999999999995
        test_mean_y = 3.275375

        expected_output = -0.11659583686778378

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_x_and_y_same(self):
        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_y = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_mean_x = 3.9944999999999995
        test_mean_y = 3.9944999999999995

        expected_output = 1.0

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_x_empty(self):
        # arrange
        test_x = []
        test_y = [7.3, -1.3, 4.1, 0.12, 0.05, 7.1, 2.12, 6.713]
        test_mean_x = None
        test_mean_y = 3.275375

        expected_output = 0.0

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_y_empty(self):
        # Expect index error, since will try to access items in empty y

        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_y = []
        test_mean_x = 3.9944999999999995
        test_mean_y = None

        # act and assert
        with self.assertRaises(IndexError):
            correlation(test_x, test_y, test_mean_x, test_mean_y)

    def test_correlation_x_zeros(self):
        # arrange
        test_x = [0.0,  0.0,  0.0,  0.0,  0.0,   0.0,    0.0,   0.0]
        test_y = [7.3, -1.3,  4.1, 0.12, 0.05,   7.1,   2.12, 6.713]
        test_mean_x = 0.0
        test_mean_y = 3.275375

        expected_output = 0.0

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_y_zeros(self):
        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_y = [0.0,  0.0,  0.0,  0.0,  0.0,   0.0,    0.0,   0.0]
        test_mean_x = 3.9944999999999995
        test_mean_y = 0.0

        expected_output = 0.0

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_x_shorter_than_y(self):
        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004]
        test_y = [7.3, -1.3,  4.1, 0.12, 0.05,   7.1,   2.12, 6.713]
        test_mean_x = 3.1923333333333326
        test_mean_y = 3.275375

        expected_output = -0.015688219407662334

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)

    def test_correlation_x_longer_than_y(self):
        # Expect IndexError since will try to access elements of y that don't exist

        # arrange
        test_x = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        test_y = [7.3, -1.3,  4.1, 0.12, 0.05,   7.1]
        test_mean_x = 3.9944999999999995
        test_mean_y = 2.8949999999999996

        # act and assert
        with self.assertRaises(IndexError):
            correlation(test_x, test_y, test_mean_x, test_mean_y)

    def test_correlation_x_and_y_each_same_num(self):
        # x all one num, y all another
        # arrange
        test_x = [ 3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0]
        test_y = [-1.3, -1.3, -1.3, -1.3, -1.3, -1.3, -1.3, -1.3]
        test_mean_x = 3.0
        test_mean_y = -1.3

        expected_output = 0.0

        # act
        output = correlation(test_x, test_y, test_mean_x, test_mean_y)

        # assert
        self.assertEqual(output, expected_output)
