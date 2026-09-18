"""Checks for dataset constraints, epoch alignment and boundary experiments."""

import math
import unittest

from src.single_neuron.boundary_experiment import ExperimentSettings, boundary, state_at
from src.single_neuron.trainable_neuron import TrainableSingleInputNeuron, train
from src.single_neuron.trainable_neuron_demo import TRAINING_INPUTS, TRAINING_TARGETS


class BoundaryExperimentTests(unittest.TestCase):
    def test_dataset_preserves_originals(self) -> None:
        """The generated pair extends rather than replaces the six examples."""
        inputs, targets = ExperimentSettings(centre=2.25, spacing=0.05).dataset()
        self.assertEqual(inputs[:6], TRAINING_INPUTS)
        self.assertEqual(targets[:6], TRAINING_TARGETS)
        self.assertEqual(inputs[6:], (2.2, 2.3))
        self.assertEqual(targets[6:], (0.0, 1.0))
        self.assertEqual(
            ExperimentSettings(augmented=False).dataset(),
            (TRAINING_INPUTS, TRAINING_TARGETS),
        )

    def test_invalid_settings(self) -> None:
        """Reject invalid geometry and unsupported training budgets."""
        for settings in (
            {"spacing": 0},
            {"centre": 2.05},
            {"centre": float("nan")},
            {"epochs": True},
            {"epochs": 20001},
            {"learning_rate": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExperimentSettings(**settings)

    def test_epoch_alignment_and_undefined_boundary(self) -> None:
        """Replay aligns pre-update records with post-update epoch numbering."""
        initial = TrainableSingleInputNeuron()
        result = train(initial, TRAINING_INPUTS, TRAINING_TARGETS, epochs=3)
        for epoch in range(4):
            neuron, evaluation = state_at(result, epoch)
            self.assertEqual(
                evaluation, neuron.evaluate(TRAINING_INPUTS, TRAINING_TARGETS)
            )
        self.assertTrue(math.isnan(boundary(initial)))
        self.assertEqual(state_at(result, 3), (result.neuron, result.final_evaluation))
        empty = train(initial, TRAINING_INPUTS, TRAINING_TARGETS, epochs=0)
        self.assertEqual(state_at(empty, 0), (initial, empty.final_evaluation))

    def test_baseline_and_endpoint_experiments(self) -> None:
        """A sufficient bounded run reaches both requested endpoint intervals."""
        baseline = train(
            TrainableSingleInputNeuron(), *ExperimentSettings(augmented=False).dataset()
        )
        self.assertAlmostEqual(boundary(baseline.neuron), 2.839667, places=5)
        for centre in (2.25, 3.75):
            with self.subTest(centre=centre):
                settings = ExperimentSettings(
                    centre=centre, spacing=0.05, learning_rate=1.0, epochs=20000
                )
                result = train(
                    TrainableSingleInputNeuron(),
                    *settings.dataset(),
                    learning_rate=settings.learning_rate,
                    epochs=settings.epochs,
                )
                self.assertLessEqual(centre - 0.05, boundary(result.neuron))
                self.assertLess(boundary(result.neuron), centre + 0.05)
                self.assertTrue(
                    all(
                        int(p > 0.5) == t
                        for p, t in zip(
                            result.final_evaluation.outputs,
                            result.final_evaluation.targets,
                        )
                    )
                )
