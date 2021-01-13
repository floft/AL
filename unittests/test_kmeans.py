from unittest import TestCase

import numpy as np

from kmeans import KMeans


class KMeansTest(TestCase):
    # Testing the KMeans model

    def test_fit_separated_data(self):
        """Test the fit on well-separated data."""

        # arrange
        test_data = [
            np.array([ 1.0,  3.0,  4.0]),  # cluster A
            np.array([ 1.0,  3.1,  4.0]),  # cluster A
            np.array([-1.0, -5.0,  0.0]),  # cluster B
            np.array([-1.1, -4.9, -0.1]),  # cluster B
            np.array([ 0.9,  2.9,  4.2]),  # cluster A
            np.array([ 1.1,  3.0,  3.9]),  # cluster A
            np.array([-1.0, -5.3,  0.2])   # cluster B
        ]

        test_num_clusters = 2

        expected_centers = np.array([
            [1.0, 3.0, 4.025],                      # cluster A (larger)
            [-1.03333333, -5.06666667, 0.03333333]  # cluster B (smaller)
        ])
        expected_n_clusters = 2

        # act
        test_kmeans = KMeans()

        output = test_kmeans.sorted_kmeans_fit(test_data, test_num_clusters)

        # assert
        # Compare returned centers to expected:
        self.assertEqual(output.shape, expected_centers.shape)

        for c_index in range(len(output)):
            actual_center = output[c_index]
            expected_center = expected_centers[c_index]

            for f_index in range(len(actual_center)):
                actual_value = actual_center[f_index]
                expected_value = expected_center[f_index]

                self.assertAlmostEqual(actual_value, expected_value, places=6)

        # Compare centers stored on kmeans to expected:
        self.assertEqual(test_kmeans.centers.shape, expected_centers.shape)

        for c_index in range(len(test_kmeans.centers)):
            actual_center = test_kmeans.centers[c_index]
            expected_center = expected_centers[c_index]

            for f_index in range(len(actual_center)):
                actual_value = actual_center[f_index]
                expected_value = expected_center[f_index]

                self.assertAlmostEqual(actual_value, expected_value, places=6)

        self.assertEqual(test_kmeans.n_clusters, expected_n_clusters)

    def test_fit_separated_data_100_times(self):
        """
        Test the fit on well-separated data. Repeat 100 times to catch any bugs due to randomness (e.g. cluster
        initialization).
        """

        for i in range(100):
            # arrange
            test_data = [
                np.array([ 1.0,  3.0,  4.0]),  # cluster A
                np.array([ 1.0,  3.1,  4.0]),  # cluster A
                np.array([-1.0, -5.0,  0.0]),  # cluster B
                np.array([-1.1, -4.9, -0.1]),  # cluster B
                np.array([ 0.9,  2.9,  4.2]),  # cluster A
                np.array([ 1.1,  3.0,  3.9]),  # cluster A
                np.array([-1.0, -5.3,  0.2])   # cluster B
            ]

            test_num_clusters = 2

            expected_centers = np.array([
                [1.0, 3.0, 4.025],                      # cluster A (larger)
                [-1.03333333, -5.06666667, 0.03333333]  # cluster B (smaller)
            ])
            expected_n_clusters = 2

            # act
            test_kmeans = KMeans()

            output = test_kmeans.sorted_kmeans_fit(test_data, test_num_clusters)

            # assert
            # Compare returned centers to expected:
            self.assertEqual(output.shape, expected_centers.shape)

            for c_index in range(len(output)):
                actual_center = output[c_index]
                expected_center = expected_centers[c_index]

                for f_index in range(len(actual_center)):
                    actual_value = actual_center[f_index]
                    expected_value = expected_center[f_index]

                    self.assertAlmostEqual(actual_value, expected_value, places=6)

            # Compare centers stored on kmeans to expected:
            self.assertEqual(test_kmeans.centers.shape, expected_centers.shape)

            for c_index in range(len(test_kmeans.centers)):
                actual_center = test_kmeans.centers[c_index]
                expected_center = expected_centers[c_index]

                for f_index in range(len(actual_center)):
                    actual_value = actual_center[f_index]
                    expected_value = expected_center[f_index]

                    self.assertAlmostEqual(actual_value, expected_value, places=6)

            self.assertEqual(test_kmeans.n_clusters, expected_n_clusters)
