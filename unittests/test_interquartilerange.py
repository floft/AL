from unittest import TestCase

from features import interquartile_range


class InterquartileRangeTest(TestCase):
    # Testing the interquartile_range method
    def test_interquartile_range_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        expected_output = 12.609

        # act
        output = interquartile_range(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_interquartile_range_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        expected_output = 0.0

        # act
        output = interquartile_range(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_interquartile_range_threedatapoints(self):
        # arrange
        test_data = [1.2, 0.05, -3.6]
        expected_output = 4.8

        # act
        output = interquartile_range(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_interquartile_range_twodatapoints(self):
        # arrange
        test_data = [1.2, -0.05]
        expected_output = 1.25

        # act
        output = interquartile_range(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_interquartile_range_onedatapoint(self):
        # arrange
        test_data = [1.2]
        expected_output = 0.0

        # act
        output = interquartile_range(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_interquartile_range_emptylist(self):
        # arrange
        test_data = []

        # act and assert
        with self.assertRaises(IndexError):
            interquartile_range(test_data)