from unittest import TestCase

from features import signal_energy


class SignalEnergyTest(TestCase):
    # Unit Tests for the signal_energy method
    def test_signal_energy_regulardata(self):
        # arrange
        test_data = [1.2, 0.05, -3.6, 18.9, -5.4, 8.004, 12.659, 0.143]
        expected_output = 625.1072459999999

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_signal_energy_allsamenum(self):
        # arrange
        test_data = [5.0, 5, 5.0, 5, 5.0000, 5, 5]
        expected_output = 175.0

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_signal_energy_allzeros(self):
        # arrange
        test_data = [0, 0.0, 000, 0.0000, 0, 0.0, 0, 000.0]
        expected_output = 0.0

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_signal_energy_allnegativenumbers(self):
        # arrange
        test_data = [-1.2, -0.05, -3.6, -18.9, -5.4, -8.004, -12.659, -0.143]
        expected_output = 625.1072459999999

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)

    def test_signal_energy_onlyonenumber(self):
        # arrange
        test_data = [3.2]
        exepected_output = 10.240000000000002

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, exepected_output)

    def test_signal_energy_emptylist(self):
        # arrange
        test_data = []
        expected_output = 0

        # act
        output = signal_energy(test_data)

        # assert
        self.assertEqual(output, expected_output)
