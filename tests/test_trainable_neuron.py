"""Verify stage-two mathematics, numerical stability and deterministic learning."""

import math
import unittest
from itertools import pairwise

import numpy as np

from src.single_neuron.trainable_neuron import (
    TrainableSingleInputNeuron,
    compare_gradients,
    sigmoid_activation,
    train,
)
from single_neuron.trainable_neuron_demo import TRAINING_INPUTS, TRAINING_TARGETS


class TrainableNeuronTests(unittest.TestCase):
    """Check explicit derivatives and the complete parameter-update trace."""

    def test_hand_calculated_batch_and_update(self) -> None:
        """At zero parameters, [0,2] with labels [0,1] gives dw=-0.5, db=0."""
        initial = TrainableSingleInputNeuron()
        result = train(initial, [0, 2], [0, 1], learning_rate=0.2, epochs=1)
        evaluation = result.steps[0].evaluation
        self.assertEqual(evaluation.outputs, (0.5, 0.5))
        self.assertEqual(evaluation.pre_activations, (0.0, 0.0))
        self.assertEqual(evaluation.logit_gradients, (0.25, -0.25))
        self.assertAlmostEqual(evaluation.loss, math.log(2))
        self.assertAlmostEqual(evaluation.weight_gradient, -0.5)
        self.assertAlmostEqual(evaluation.bias_gradient, 0.0)
        self.assertAlmostEqual(result.neuron.weight, 0.1)
        self.assertAlmostEqual(result.neuron.bias, 0.0)
        self.assertEqual(initial, TrainableSingleInputNeuron())
        self.assertEqual(
            result.final_evaluation, result.neuron.evaluate([0, 2], [0, 1])
        )

    def test_inspection_matches_batch_forward(self) -> None:
        """Stage-one scalar inspection and NumPy batch calculations must agree."""
        neuron = TrainableSingleInputNeuron(2.0, -6.0)
        evaluation = neuron.evaluate([2, 3, 4], [0, 0, 1])
        for index, x in enumerate([2, 3, 4]):
            inspected = neuron.inspect(x)
            self.assertEqual(
                inspected.weighted_input, evaluation.weighted_inputs[index]
            )
            self.assertEqual(
                inspected.pre_activation, evaluation.pre_activations[index]
            )
            self.assertAlmostEqual(inspected.output, evaluation.outputs[index])
        self.assertEqual(neuron.forward(3), 0.5)

    def test_gradients_match_central_differences(self) -> None:
        """Both gradients agree away from the symmetric zero initial state."""
        for weight, bias in [(0.0, 0.0), (0.7, -0.4), (-1.2, 0.8)]:
            with self.subTest(weight=weight, bias=bias):
                neuron = TrainableSingleInputNeuron(weight, bias)
                comparison = compare_gradients(neuron, [-2, 0.5, 3], [1, 0, 1])
                np.testing.assert_allclose(
                    [comparison.analytical_weight, comparison.analytical_bias],
                    [comparison.numerical_weight, comparison.numerical_bias],
                    atol=1e-7,
                    rtol=1e-5,
                )
                self.assertEqual(neuron, TrainableSingleInputNeuron(weight, bias))

    def test_learns_threshold_and_records_every_update(self) -> None:
        """Default training lowers loss, separates the labels and exposes updates."""
        result = train(TrainableSingleInputNeuron(), TRAINING_INPUTS, TRAINING_TARGETS)
        losses = [step.evaluation.loss for step in result.steps]
        losses.append(result.final_evaluation.loss)
        self.assertEqual(len(result.steps), 2000)
        self.assertTrue(all(after < before for before, after in pairwise(losses)))
        self.assertLess(losses[-1], 0.1)
        threshold = -result.neuron.bias / result.neuron.weight
        self.assertAlmostEqual(threshold, 3.0, delta=0.2)
        self.assertEqual(
            tuple(float(p > 0.5) for p in result.final_evaluation.outputs),
            TRAINING_TARGETS,
        )
        for index, step in enumerate(result.steps):
            self.assertEqual(step.epoch, index + 1)
            self.assertEqual(
                step.after.weight,
                step.before.weight
                - step.learning_rate * step.evaluation.weight_gradient,
            )
            self.assertEqual(
                step.after.bias,
                step.before.bias - step.learning_rate * step.evaluation.bias_gradient,
            )
            if index:
                self.assertEqual(step.before, result.steps[index - 1].after)

    def test_extreme_logits_remain_finite(self) -> None:
        """Confidently wrong predictions have large finite loss and useful gradients."""
        neuron = TrainableSingleInputNeuron(1000, 0)
        evaluation = neuron.evaluate([-1, 1], [1, 0])
        self.assertEqual(evaluation.loss, 1000)
        self.assertEqual(evaluation.weight_gradient, 1)
        self.assertEqual(sigmoid_activation(-1000), 0)
        self.assertEqual(sigmoid_activation(1000), 1)

    def test_rejects_invalid_data(self) -> None:
        """Reject empty, mismatched, multidimensional and non-finite training data."""
        neuron = TrainableSingleInputNeuron()
        for inputs, targets in [
            ([], []),
            ([1], []),
            ([[1]], [[0]]),
            ([math.nan], [0]),
            ([math.inf], [1]),
            ([1], [math.nan]),
            ([1], [0.5]),
            ([1], [2]),
        ]:
            with (
                self.subTest(inputs=inputs, targets=targets),
                self.assertRaises(ValueError),
            ):
                neuron.evaluate(inputs, targets)

    def test_rejects_invalid_configuration(self) -> None:
        """Invalid parameters and training/checking settings fail explicitly."""
        for value in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                TrainableSingleInputNeuron(value, 0)
            with self.assertRaises(ValueError):
                TrainableSingleInputNeuron(0, value)
            with self.assertRaises(ValueError):
                sigmoid_activation(value)
        for value in [0, -0.1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                train(TrainableSingleInputNeuron(), [1], [0], learning_rate=value)
            with self.assertRaises(ValueError):
                compare_gradients(TrainableSingleInputNeuron(), [1], [0], epsilon=value)
        for epochs in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                train(TrainableSingleInputNeuron(), [1], [0], epochs=epochs)

    def test_zero_epochs_preserves_initial_state(self) -> None:
        """Zero updates still evaluates the supplied dataset."""
        initial = TrainableSingleInputNeuron(0.2, -0.1)
        result = train(initial, [1], [0], epochs=0)
        self.assertEqual(result.neuron, initial)
        self.assertEqual(result.steps, ())
        self.assertEqual(result.final_evaluation, initial.evaluate([1], [0]))


if __name__ == "__main__":
    unittest.main()
