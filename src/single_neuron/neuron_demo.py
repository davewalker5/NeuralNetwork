"""Run an inspectable demonstration of a single-input neuron."""

from .neuron import SingleInputNeuron

DEFAULT_WEIGHT = 2.0
DEFAULT_BIAS = -6.0
DEMONSTRATION_INPUTS = (0.0, 2.0, 3.0, 4.0, 6.0)


def format_forward_pass(neuron: SingleInputNeuron, input_value: float) -> str:
    """
    Format the complete calculation for one input as a readable line.

    :param neuron: Neuron used to calculate the output.
    :param input_value: Value supplied to the neuron.
    :return: Formatted calculation and result.
    """
    result = neuron.inspect(input_value)
    return (
        f"x={result.input_value:>4.1f}  "
        f"w*x={result.weighted_input:>5.1f}  "
        f"b={result.bias:>5.1f}  "
        f"z={result.pre_activation:>5.1f}  "
        f"step(z)={result.output:.0f}"
    )


def main() -> None:
    """Display how a fixed single-input neuron processes several inputs."""
    neuron = SingleInputNeuron(weight=DEFAULT_WEIGHT, bias=DEFAULT_BIAS)

    print("Single-input artificial neuron")
    print(f"z = ({neuron.weight:.1f} * x) + ({neuron.bias:.1f})")
    print("activation = 1 when z > 0, otherwise 0")
    print()

    for input_value in DEMONSTRATION_INPUTS:
        print(format_forward_pass(neuron, input_value))


if __name__ == "__main__":
    main()
