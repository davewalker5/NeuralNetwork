"""Tests for the single-input artificial neuron."""

import math
import unittest

from src.single_neuron import SingleInputNeuron, step_activation


class StepActivationTests(unittest.TestCase):
    """Verify the step activation around its zero threshold."""

    def test_returns_zero_at_and_below_zero(self) -> None:
        """The step activation should be off until its input exceeds zero."""
        self.assertEqual(step_activation(-0.1), 0.0)
        self.assertEqual(step_activation(0.0), 0.0)

    def test_returns_one_above_zero(self) -> None:
        """The step activation should be on for a positive input."""
        self.assertEqual(step_activation(0.1), 1.0)


class SingleInputNeuronTests(unittest.TestCase):
    """Verify forward calculations and validation for a single neuron."""

    def setUp(self) -> None:
        """Create the fixed neuron used by the threshold examples."""
        self.neuron = SingleInputNeuron(weight=2.0, bias=-6.0)

    def test_output_is_zero_below_threshold(self) -> None:
        """The neuron should be off when the input is below three."""
        self.assertEqual(self.neuron.forward(2.0), 0.0)

    def test_output_is_zero_at_threshold(self) -> None:
        """The neuron should be off when its pre-activation is exactly zero."""
        self.assertEqual(self.neuron.forward(3.0), 0.0)

    def test_output_is_one_above_threshold(self) -> None:
        """The neuron should be on when the input is above three."""
        self.assertEqual(self.neuron.forward(4.0), 1.0)

    def test_inspect_exposes_forward_pass_values(self) -> None:
        """Inspection should expose each value in the neuron calculation."""
        result = self.neuron.inspect(4.0)

        self.assertEqual(result.input_value, 4.0)
        self.assertEqual(result.weighted_input, 8.0)
        self.assertEqual(result.bias, -6.0)
        self.assertEqual(result.pre_activation, 2.0)
        self.assertEqual(result.output, 1.0)

    def test_custom_activation_is_used(self) -> None:
        """A supplied activation function should receive the calculated value."""
        identity_neuron = SingleInputNeuron(
            weight=2.0,
            bias=-6.0,
            activation=lambda value: value,
        )

        self.assertEqual(identity_neuron(4.0), 2.0)

    def test_rejects_non_finite_parameters(self) -> None:
        """Weights and biases must not contain infinities or NaN values."""
        with self.assertRaises(ValueError):
            SingleInputNeuron(weight=math.inf, bias=0.0)

        with self.assertRaises(ValueError):
            SingleInputNeuron(weight=1.0, bias=math.nan)

    def test_rejects_non_finite_input(self) -> None:
        """Forward passes must reject non-finite input values."""
        with self.assertRaises(ValueError):
            self.neuron.forward(math.nan)


if __name__ == "__main__":
    unittest.main()
