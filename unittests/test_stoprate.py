from unittest import TestCase

from features import stop_rate


class StopRateTest(TestCase):
    # Unit tests for the caclkstoprate method

    def test_stop_rate_alldifferentdata(self):
        # arrange
        test_lat = [46.7194, 46.7245, 46.7592, 46.7290, 46.7500, 46.4972, 46.3946, 46.1074]
        test_long = [-117.184, -117.205, -117.163, -117.211, -117.156, -117.134, -117.199, -117.153]
        test_distance = 0.0005
        expected_output = 2000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latandlongallsamenum(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005
        expected_output = 16000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latchangeslongstayssame(self):
        # arrange
        test_lat = [46.7194, 46.7245, 46.7592, 46.7290, 46.7500, 46.4972, 46.3946, 46.1074]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005
        expected_output = 2000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latstayssamelongchanges(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.205, -117.163, -117.211, -117.156, -117.134, -117.199, -117.153]
        test_distance = 0.0005
        expected_output = 2000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_bothvarythenstopthenvaryagain(self):
        # arrange
        test_lat = [46.7194, 46.7245, 46.7592, 46.7290, 46.7290, 46.7290, 46.3946, 46.1074]
        test_long = [-117.184, -117.205, -117.163, -117.211, -117.211, -117.211, -117.199, -117.153]
        test_distance = 0.0005
        expected_output = 6000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latisallzeroslongallsamenum(self):
        # arrange
        test_lat = [0.000, 0.0, 0, 0.0, 0000.0, 00.0, 0.0, 0.0]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005
        expected_output = 16000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latallsamenumlongallzeros(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [0.000, 0.0, 0, 0.0, 0000.0, 00.0, 0.0, 0.0]
        test_distance = 0.0005
        expected_output = 16000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latemptylist(self):
        # arrange
        test_lat = []
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005

        # act and assert
        with self.assertRaises(IndexError):
            stop_rate(test_lat, test_long, test_distance)

    def test_stop_rate_longemptylist(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = []
        test_distance = 0.0005

        # act and assert
        with self.assertRaises(IndexError):
            stop_rate(test_lat, test_long, test_distance)

    def test_stop_rate_distanceiszero(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0
        expected_output = 0.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_latislongerthanlong(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005

        # act and assert
        with self.assertRaises(IndexError):
            stop_rate(test_lat, test_long, test_distance)

    def test_stop_rate_longislongerthanlat(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 0.0005
        expected_output = 14000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_twogpspoints(self):
        # arrange
        test_lat = [46.7194, 46.7194]
        test_long = [-117.184, -117.184]
        test_distance = 0.0005
        expected_output = 4000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_onegpspoint(self):
        # arrange
        test_lat = [46.7194]
        test_long = [-117.184]
        test_distance = 0.0005
        expected_output = 2000.0

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)

    def test_stop_rate_distanceislarge(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        test_distance = 529.37
        expected_output = 0.01511230330392731

        # act
        output = stop_rate(test_lat, test_long, test_distance)

        # assert
        self.assertEqual(output, expected_output)
