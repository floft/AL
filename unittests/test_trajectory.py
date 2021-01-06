from unittest import TestCase

from features import trajectory


class TrajectoryTest(TestCase):
    # Unit tests for the trajectory method

    def test_trajectory_allsamenumbers(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_alldifferentnumbers(self):
        # arrange
        test_lat = [46.7194, 46.7245, 46.7592, 46.7290, 46.7500, 46.4972, 46.3946, 46.1074]
        test_long = [-117.184, -117.205, -117.163, -117.211, -117.156, -117.134, -117.199, -117.153]
        expected_output = 0.11758940249719632

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latchangeslongstayssame(self):
        # arrange
        test_lat = [46.7194, 46.7245, 46.7592, 46.7290, 46.7500, 46.4972, 46.3946, 46.1074]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        expected_output = 0.0

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latstayssamelongchanges(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.205, -117.163, -117.211, -117.156, -117.134, -117.199, -117.153]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latallzeroslongallsamenum(self):
        # arrange
        test_lat = [0.000, 0.0, 0, 0.0, 0000.0, 00.0, 0.0, 0.0]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latallsamenumlongallzeros(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [0.000, 0.0, 0, 0.0, 0000.0, 00.0, 0.0, 0.0]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latemtpylist(self):
        # arrange
        test_lat = []
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]

        # act and assert
        with self.assertRaises(ValueError):
            trajectory(test_lat, test_long)

    def test_trajectory_longemptylist(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = []
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_latislongerthanlong(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_longislongerthanlat(self):
        # arrange
        test_lat = [46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194, 46.7194]
        test_long = [-117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184, -117.184]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)

    def test_trajectory_onegpspoint(self):
        # arrange
        test_lat = [46.7194]
        test_long = [-117.184]
        expected_output = -1.57079633

        # act
        output = trajectory(test_lat, test_long)

        # assert
        self.assertEqual(output, expected_output)
