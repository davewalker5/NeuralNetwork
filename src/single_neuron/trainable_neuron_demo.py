"""Run stage two with a deterministic dataset and an inspectable training trace."""

from .trainable_neuron import (
    TrainableSingleInputNeuron,
    compare_gradients,
    train,
)

# Labels are zero below three and one above three; omit the ambiguous boundary.
TRAINING_INPUTS = (0.0, 1.0, 2.0, 4.0, 5.0, 6.0)
TRAINING_TARGETS = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def main() -> None:
    """Display gradient checks, sampled updates and learned predictions."""
    initial = TrainableSingleInputNeuron()
    comparison = compare_gradients(initial, TRAINING_INPUTS, TRAINING_TARGETS)
    print("Trainable single-input neuron: sigmoid(w*x + b)")
    print("Mean binary cross-entropy; full-batch gradient descent; learning rate=0.1")
    print("Gradient check: epsilon=1e-5, absolute tolerance=1e-7, relative=1e-5")
    print(
        f"weight: analytical={comparison.analytical_weight:.10f}, "
        f"numerical={comparison.numerical_weight:.10f}"
    )
    print(
        f"bias:   analytical={comparison.analytical_bias:.10f}, "
        f"numerical={comparison.numerical_bias:.10f}"
    )
    result = train(initial, TRAINING_INPUTS, TRAINING_TARGETS)
    print("\nPre-update loss and gradients; w_next=w-rate*dw, b_next=b-rate*db")
    print("epoch      loss         w         b        dw        db    w_next    b_next")
    for step in result.steps:
        if step.epoch == 1 or step.epoch % 200 == 0:
            print(
                f"{step.epoch:5d} {step.evaluation.loss:9.6f} "
                f"{step.before.weight:9.5f} {step.before.bias:9.5f} "
                f"{step.evaluation.weight_gradient:9.5f} "
                f"{step.evaluation.bias_gradient:9.5f} "
                f"{step.after.weight:9.5f} {step.after.bias:9.5f}"
            )
    print(f"\nFinal loss: {result.final_evaluation.loss:.6f}")
    print(f"Learned threshold (-b/w): {-result.neuron.bias / result.neuron.weight:.4f}")
    print("Class one when probability > 0.5; at 0.5 choose class zero.")
    for x, target in zip(TRAINING_INPUTS, TRAINING_TARGETS):
        probability = result.neuron.forward(x)
        print(
            f"x={x:.1f} target={target:.0f} probability={probability:.4f} "
            f"class={int(probability > 0.5)}"
        )


if __name__ == "__main__":
    main()
