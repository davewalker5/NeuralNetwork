"""A small, inspectable implementation of a single-input neuron."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

ActivationFunction = Callable[[float], float]


def step_activation(value: float) -> float:
    """
    Return a binary output using zero as the activation threshold.

    :param value: Pre-activation value produced by the neuron.
    :return: One when ``value`` is greater than zero; otherwise zero.
    """
    return 1.0 if value > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class ForwardPass:
    """Intermediate and final values produced by one neuron calculation."""

    input_value: float
    weighted_input: float
    bias: float
    pre_activation: float
    output: float


@dataclass(frozen=True, slots=True)
class SingleInputNeuron:
    """An artificial neuron with one input, one weight and one bias."""

    weight: float
    bias: float
    activation: ActivationFunction = step_activation

    def __post_init__(self) -> None:
        """
        Validate the neuron parameters after initialisation.

        :raises ValueError: If the weight or bias is not finite.
        :raises TypeError: If the activation is not callable.
        """
        if not isfinite(self.weight):
            raise ValueError("weight must be a finite number")
        if not isfinite(self.bias):
            raise ValueError("bias must be a finite number")
        if not callable(self.activation):
            raise TypeError("activation must be callable")

    def calculate_pre_activation(self, input_value: float) -> float:
        """
        Calculate the weighted input plus the bias.

        This is the neuron's linear calculation ``z = (w * x) + b``.

        :param input_value: Single numeric input to the neuron.
        :return: The pre-activation value ``z``.
        :raises ValueError: If the input is not finite.
        """
        if not isfinite(input_value):
            raise ValueError("input_value must be a finite number")

        return (self.weight * input_value) + self.bias

    def forward(self, input_value: float) -> float:
        """
        Pass an input through the neuron and return its output.

        :param input_value: Single numeric input to the neuron.
        :return: Output returned by the activation function.
        :raises ValueError: If the input is not finite.
        """
        pre_activation = self.calculate_pre_activation(input_value)
        return self.activation(pre_activation)

    def inspect(self, input_value: float) -> ForwardPass:
        """
        Return every value involved in a forward pass for inspection.

        :param input_value: Single numeric input to the neuron.
        :return: A record containing the forward-pass calculation.
        :raises ValueError: If the input is not finite.
        """
        pre_activation = self.calculate_pre_activation(input_value)

        return ForwardPass(
            input_value=input_value,
            weighted_input=self.weight * input_value,
            bias=self.bias,
            pre_activation=pre_activation,
            output=self.activation(pre_activation),
        )

    def __call__(self, input_value: float) -> float:
        """
        Pass an input through the neuron using callable syntax.

        :param input_value: Single numeric input to the neuron.
        :return: Output returned by the activation function.
        """
        return self.forward(input_value)
