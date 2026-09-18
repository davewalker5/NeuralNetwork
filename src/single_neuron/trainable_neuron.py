"""Train a single-input sigmoid neuron using explicit NumPy gradients."""

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from numpy.typing import ArrayLike

from .neuron import ForwardPass, SingleInputNeuron


def sigmoid_activation(value: float) -> float:
    """
    Calculate sigmoid without overflowing at large negative inputs.

    :param value: Finite pre-activation value.
    :return: Probability between zero and one.
    :raises ValueError: If the input is not finite.
    """
    if not isfinite(value):
        raise ValueError("value must be finite")
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class BatchEvaluation:
    """Forward values, mean loss and derivatives for one labelled batch.

    Each tuple follows the input order. ``logit_gradients`` contains dL/dz
    including the factor 1/n from the mean loss.
    """

    inputs: tuple[float, ...]
    targets: tuple[float, ...]
    weighted_inputs: tuple[float, ...]
    pre_activations: tuple[float, ...]
    outputs: tuple[float, ...]
    sample_losses: tuple[float, ...]
    logit_gradients: tuple[float, ...]
    loss: float
    weight_gradient: float
    bias_gradient: float


@dataclass(frozen=True, slots=True)
class TrainableSingleInputNeuron:
    """One weight and bias, trained by returning updated immutable neurons."""

    weight: float = 0.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        """Validate finite parameters using the stage-one neuron."""
        SingleInputNeuron(self.weight, self.bias)

    def inspect(self, input_value: float) -> ForwardPass:
        """
        Reuse stage one's calculation with a differentiable activation.

        :param input_value: Single finite numeric input.
        :return: Weighted input, bias, pre-activation and sigmoid output.
        :raises ValueError: If the input or computed logit is not finite.
        """
        return SingleInputNeuron(self.weight, self.bias, sigmoid_activation).inspect(
            input_value
        )

    def forward(self, input_value: float) -> float:
        """
        Predict a probability for one input.

        :param input_value: Single finite numeric input.
        :return: Sigmoid probability of class one.
        """
        return self.inspect(input_value).output

    def evaluate(self, inputs: ArrayLike, targets: ArrayLike) -> BatchEvaluation:
        """
        Calculate a forward pass, mean binary cross-entropy and gradients.

        :param inputs: Nonempty one-dimensional finite input sequence.
        :param targets: Matching sequence of binary labels (zero or one).
        :return: Every per-example value and both parameter gradients.
        :raises ValueError: If the data is invalid or calculations overflow.
        """
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 1 or x.size == 0 or y.shape != x.shape:
            raise ValueError("inputs and targets must be matching nonempty 1D arrays")
        if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("inputs must be finite and targets must be zero or one")

        with np.errstate(over="raise", invalid="raise"):
            try:
                weighted_inputs = self.weight * x
                z = weighted_inputs + self.bias
                # exp(-abs(z)) is safe even for confidently wrong predictions.
                exponential = np.exp(-np.abs(z))
                probabilities = np.where(
                    z >= 0, 1 / (1 + exponential), exponential / (1 + exponential)
                )
                # BCE from logits avoids log(0) and cancellation at large z.
                sample_losses = np.logaddexp(0.0, np.where(y == 1, -z, z))
                loss = float(np.mean(sample_losses))
                # dL/dz = (sigmoid(z) - y)/n: sigmoid and BCE simplify together.
                logit_gradients = (probabilities - y) / x.size
                weight_gradient = float(np.sum(logit_gradients * x))
                bias_gradient = float(np.sum(logit_gradients))
            except FloatingPointError as error:
                raise ValueError(
                    "batch calculation overflowed; use smaller values"
                ) from error

        return BatchEvaluation(
            inputs=tuple(x.tolist()),
            targets=tuple(y.tolist()),
            weighted_inputs=tuple(weighted_inputs.tolist()),
            pre_activations=tuple(z.tolist()),
            outputs=tuple(probabilities.tolist()),
            sample_losses=tuple(sample_losses.tolist()),
            logit_gradients=tuple(logit_gradients.tolist()),
            loss=loss,
            weight_gradient=weight_gradient,
            bias_gradient=bias_gradient,
        )


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """A complete update: pre-update state, evaluation, rate and new state."""

    epoch: int
    before: TrainableSingleInputNeuron
    evaluation: BatchEvaluation
    learning_rate: float
    after: TrainableSingleInputNeuron


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Trained neuron and every update, including the final evaluation."""

    neuron: TrainableSingleInputNeuron
    steps: tuple[TrainingStep, ...]
    final_evaluation: BatchEvaluation


def train(
    neuron: TrainableSingleInputNeuron,
    inputs: ArrayLike,
    targets: ArrayLike,
    learning_rate: float = 0.1,
    epochs: int = 2000,
) -> TrainingResult:
    """
    Perform full-batch gradient descent without modifying the initial neuron.

    :param neuron: Known initial weight and bias.
    :param inputs: One-dimensional finite training inputs.
    :param targets: Matching binary labels.
    :param learning_rate: Positive finite update multiplier.
    :param epochs: Nonnegative integer number of parameter updates.
    :return: Final neuron, complete trace and loss after the final update.
    :raises ValueError: If configuration, data or updated parameters are invalid.
    """
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    evaluation = neuron.evaluate(inputs, targets)
    steps = []
    for epoch in range(1, epochs + 1):
        updated = TrainableSingleInputNeuron(
            weight=neuron.weight - learning_rate * evaluation.weight_gradient,
            bias=neuron.bias - learning_rate * evaluation.bias_gradient,
        )
        steps.append(TrainingStep(epoch, neuron, evaluation, learning_rate, updated))
        neuron = updated
        evaluation = neuron.evaluate(inputs, targets)
    return TrainingResult(neuron, tuple(steps), evaluation)


@dataclass(frozen=True, slots=True)
class GradientComparison:
    """Analytical and central finite-difference derivatives for both parameters."""

    analytical_weight: float
    numerical_weight: float
    analytical_bias: float
    numerical_bias: float


def compare_gradients(
    neuron: TrainableSingleInputNeuron,
    inputs: ArrayLike,
    targets: ArrayLike,
    epsilon: float = 1e-5,
) -> GradientComparison:
    """
    Independently approximate each derivative with (L(p+h)-L(p-h))/(2h).

    Use absolute tolerance 1e-7 and relative tolerance 1e-5 on modest logits.
    Very large parameters or tiny perturbations can lose floating-point precision.

    :param neuron: State at which to compare gradients; remains unchanged.
    :param inputs: One-dimensional finite inputs.
    :param targets: Matching binary labels.
    :param epsilon: Positive finite parameter perturbation.
    :return: Analytical and numerical gradients of the same mean loss.
    :raises ValueError: If epsilon or evaluation data is invalid.
    """
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    analytical = neuron.evaluate(inputs, targets)
    weight_plus = TrainableSingleInputNeuron(neuron.weight + epsilon, neuron.bias)
    weight_minus = TrainableSingleInputNeuron(neuron.weight - epsilon, neuron.bias)
    bias_plus = TrainableSingleInputNeuron(neuron.weight, neuron.bias + epsilon)
    bias_minus = TrainableSingleInputNeuron(neuron.weight, neuron.bias - epsilon)
    return GradientComparison(
        analytical.weight_gradient,
        (
            weight_plus.evaluate(inputs, targets).loss
            - weight_minus.evaluate(inputs, targets).loss
        )
        / (2 * epsilon),
        analytical.bias_gradient,
        (
            bias_plus.evaluate(inputs, targets).loss
            - bias_minus.evaluate(inputs, targets).loss
        )
        / (2 * epsilon),
    )
