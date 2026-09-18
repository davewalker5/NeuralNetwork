"""Dataset generation and trace inspection for the boundary explorer."""

from dataclasses import dataclass
from math import isfinite

from .trainable_neuron import (
    BatchEvaluation,
    TrainableSingleInputNeuron,
    TrainingResult,
)
from .trainable_neuron_demo import TRAINING_INPUTS, TRAINING_TARGETS

MIN_SPACING = 0.01
MAX_EPOCHS = 20000
BOUNDARY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """A bounded, reproducible experiment; spacing is the pair's half-spacing."""

    centre: float = 3.0
    spacing: float = 0.1
    augmented: bool = True
    learning_rate: float = 0.1
    epochs: int = 2000

    def __post_init__(self) -> None:
        """Require a separated pair strictly inside the original gap."""
        if not all(
            isfinite(v) for v in (self.centre, self.spacing, self.learning_rate)
        ):
            raise ValueError("settings must be finite")
        if not MIN_SPACING <= self.spacing <= 0.5:
            raise ValueError("half-spacing must be between 0.01 and 0.5")
        if not 2 < self.centre - self.spacing < self.centre + self.spacing < 4:
            raise ValueError(
                "both additional examples must lie strictly between 2 and 4"
            )
        if not 0.001 <= self.learning_rate <= 1:
            raise ValueError("learning rate must be between 0.001 and 1")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or not 1 <= self.epochs <= MAX_EPOCHS
        ):
            raise ValueError("epochs must be an integer between 1 and 20000")

    def dataset(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Retain the originals and optionally append exactly one labelled pair.

        :return: Input and target tuples in the original order, then the pair.
        """
        if not self.augmented:
            return TRAINING_INPUTS, TRAINING_TARGETS
        return (
            TRAINING_INPUTS + (self.centre - self.spacing, self.centre + self.spacing),
            TRAINING_TARGETS + (0.0, 1.0),
        )


def boundary(neuron: TrainableSingleInputNeuron) -> float:
    """Return NaN for non-finite thresholds or weights within 1e-12 of zero.

    :param neuron: Recorded neuron state to inspect.
    :return: Input at probability 0.5, or NaN if undefined for display.
    """
    if abs(neuron.weight) <= BOUNDARY_TOLERANCE:
        return float("nan")
    value = -neuron.bias / neuron.weight
    return value if isfinite(value) else float("nan")


def state_at(
    result: TrainingResult, epoch: int
) -> tuple[TrainableSingleInputNeuron, BatchEvaluation]:
    """Select matching parameters and evaluation AFTER the requested update.

    :param result: Complete recorded training run.
    :param epoch: Number of updates, including zero for the initial state.
    :return: Matching neuron and batch evaluation.
    :raises ValueError: If the epoch is outside the trace.
    """
    if not 0 <= epoch <= len(result.steps):
        raise ValueError("epoch is outside the recorded run")
    if epoch == len(result.steps):
        return result.neuron, result.final_evaluation
    step = result.steps[epoch]
    return step.before, step.evaluation
