# Stage 2: a trainable single-input neuron

Stage one remains in `neuron.py` and `demo.py`, unchanged. Stage two lives in
`trainable_neuron.py` and `trainable_neuron_demo.py`. It reuses the original neuron's
scalar forward-pass inspection, replacing the step activation with sigmoid.
NumPy makes batch calculations and manually calculated gradients explicit.

## Run

From the project root, install the declared dependency into the virtual environment:

```sh
venv/bin/python -m pip install 'numpy>=2.3,<3'
venv/bin/python -m src.single_neuron.trainable_neuron_demo
venv/bin/python -m unittest discover -v
```

The original demo still runs with `venv/bin/python -m src.single_neuron.demo`.

## Mathematics

For each example with input x and binary target y:

- Weighted input: w*x; pre-activation (logit): z = w*x + b.
- Probability: p = sigmoid(z) = 1/(1 + exp(-z)).
- Sigmoid derivative: dp/dz = p*(1-p).
- Binary cross-entropy: loss = -y*log(p) - (1-y)*log(1-p).
- Loss derivative: d(loss)/dp = -y/p + (1-y)/(1-p).

Applying the chain rule simplifies their product to d(loss)/dz = p-y.
For a batch of n examples, we minimise **mean** loss:

```text
dL/dz_i = (p_i-y_i)/n
dL/dw   = sum(dL/dz_i * x_i)
dL/db   = sum(dL/dz_i)
w_next  = w - learning_rate * dL/dw
b_next  = b - learning_rate * dL/db
```

The implementation evaluates BCE directly from logits with `logaddexp(0, -z)`
for label one and `logaddexp(0, z)` for label zero. This is the same loss but
avoids taking log(0) when probabilities round to zero or one. Sigmoid uses
exp(-abs(z)) to avoid exponential overflow. There is no automatic differentiation.

### Check by hand

Start at w=0 and b=0 with inputs [0, 2] and labels [0, 1]. Both probabilities
are 0.5, and the mean loss is log(2), approximately 0.693147.
The two dL/dz values are 0.25 and -0.25, so dL/dw=-0.5 and dL/db=0.
With learning rate 0.2, the updated parameters are w=0.1 and b=0.
The tests verify these numbers.

## Dataset and expected behaviour

The demo uses inputs [0, 1, 2, 4, 5, 6] with labels [0, 0, 0, 1, 1, 1].
The intended threshold is three; there is no training example exactly at three.
Initial parameters are both zero. With learning rate 0.1 and 2,000 full-batch
updates, loss decreases from 0.693147 to approximately 0.035555, and the learned
threshold -b/w is approximately 2.8397. All six training labels are classified
correctly. The data only constrains the boundary to fall between two and four;
it does not uniquely identify three.

Classification chooses one when p > 0.5, otherwise zero, matching stage one's
strict z > 0 convention. For positive w, class one lies above -b/w; for negative
w, it lies below it. A zero weight has no input-dependent threshold. On perfectly
separable data, unregularised BCE can keep decreasing as parameters grow; this
example intentionally uses a fixed number of updates rather than claiming a
finite optimal weight.

## Inspect or customise training

```python
from src.single_neuron.trainable_neuron import TrainableSingleInputNeuron, train

initial = TrainableSingleInputNeuron(weight=0.0, bias=0.0)
result = train(initial, [0, 1, 2, 4, 5, 6], [0, 0, 0, 1, 1, 1],
               learning_rate=0.1, epochs=2000)
first = result.steps[0]
# first.before, first.evaluation, first.learning_rate, first.after
# first.evaluation includes each input, target, weighted input, logit,
# probability, sample loss and dL/dz, plus mean loss and both gradients.
# result.final_evaluation measures the model AFTER the last update.
```

Training returns new immutable neuron states, leaving the initial neuron intact.
Every update is retained, while the demo prints only the first and every 200th
update for readability. Trace memory grows with examples times epochs; this is
intended for small educational datasets. Initial values, data, learning rate and
update count are caller-controlled, with no randomness. Large learning rates
can make loss increase.

## Numerical gradient check

`compare_gradients` perturbs one parameter at a time, holding the other fixed:

```text
numerical derivative = (L(parameter + epsilon) - L(parameter - epsilon))
                       / (2 * epsilon)
```

Use epsilon=1e-5 and compare with absolute tolerance 1e-7 plus relative tolerance
1e-5 (`numpy.testing.assert_allclose`). Tests check both derivatives at zero and
at nonzero parameters on an asymmetric dataset, so the bias check is nontrivial.
These tolerances apply to the modest logits in these examples; extreme scales
or tiny epsilon can cause subtraction to lose precision. The check leaves the
original neuron unchanged.
