from unittest import TestCase

from features import log_signal_energy


class LogSignalEnergyTest(TestCase):
    # Unit tests for the log_signal_energy method
    def test_log_signal_energy_relgulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        expected_output = 5.008703664882724

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        expected_output = 9.785580060704262

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        expected_output = 0

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_emptylist(self):
        # arrange
        test_data = []
        expected_output = 0

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_allnegativenumbers(self):
        # arrange
        test_data = [-1.2, -0.05, -3.6, -18.9, -5.4, -8.004, -12.659, -0.143]
        expected_output = 5.008703664882724

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_onlyonenumber(self):
        # arrange
        test_data = [3.2]
        expected_output = 1.0102999566398119

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_log_signal_energy_allones(self):
        # arrange
        test_data = [1.0, 1, 1.0, 1, 1.0000, 1, 1]
        expected_output = 0.0

        # act
        output = log_signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)
