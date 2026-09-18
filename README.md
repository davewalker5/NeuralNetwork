[![Build Status](https://github.com/davewalker5/NeuralNetwork/workflows/Python%20CI%20Build/badge.svg)](https://github.com/davewalker5/NeuralNetwork/actions)
[![Coverage](https://codecov.io/gh/davewalker5/NeuralNetwork/branch/main/graph/badge.svg?token=U86UFDVD5S)](https://codecov.io/gh/davewalker5/NeuralNetwork)
[![GitHub issues](https://img.shields.io/github/issues/davewalker5/NeuralNetwork)](https://github.com/davewalker5/NeuralNetwork/issues)
[![Releases](https://img.shields.io/github/v/release/davewalker5/NeuralNetwork.svg?include_prereleases)](https://github.com/davewalker5/NeuralNetwork/releases)
[![License: MIT](https://img.shields.io/badge/License-mit-blue.svg)](https://github.com/davewalker5/NeuralNetwork/blob/main/LICENSE)
[![Language](https://img.shields.io/badge/language-python-blue.svg)](https://www.python.org)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/davewalker5/NeuralNetwork)](https://github.com/davewalker5/NeuralNetwork/)

# Neural Network Exploration

An investigative and educational project exploring how neural networks work, beginning with the smallest useful artificial neuron and progressing towards a tiny GPT-style language model.

The aim is **understanding rather than model performance**. The project deliberately keeps the underlying machinery visible so that weights, biases, activations, loss calculations, gradients and parameter updates can be inspected rather than hidden behind high-level machine-learning APIs.

## The Journey

The project follows a staged progression:

1. **Single-input neuron** — explore weights, biases and activation functions using a model simple enough to calculate by hand
2. **Trainable neuron** — introduce loss, gradients and gradient descent
3. **Two-input neuron** — move from a threshold to a visible decision boundary
4. **Multilayer network** — combine neurons and implement backpropagation explicitly
5. **Inspect learning** — observe losses, parameters, activations and gradients as training takes place
6. **Tiny language model** — build a small GPT-style transformer and connect its operation back to the same principles introduced by the first neuron

Each stage is intended to remain small enough to understand and runnable in isolation.

## Approach

The foundational models use **Python and NumPy**, with the important calculations implemented explicitly rather than delegated to a neural-network framework.

The project favours:

* Small, deterministic examples
* Mathematics that can be independently checked
* Visible intermediate values and diagnostic output
* Plots where they help explain model behaviour
* Manually implemented forward propagation, gradients and backpropagation in the foundational stages
* Models and datasets small enough to explore comfortably on a laptop

**PyTorch** may be introduced for the transformer stage, where tensor operations and automatic differentiation make the architecture practical, while retaining an explicit connection to the operations implemented manually earlier in the project.

## From One Neuron to a Transformer

The central idea behind the project is that the later models should feel like a continuation of the earlier ones rather than an unrelated black box:

```text
input
  ↓
weight + bias
  ↓
activation
  ↓
multiple inputs
  ↓
multiple neurons
  ↓
multiple layers
  ↓
backpropagation
  ↓
embeddings and attention
  ↓
tiny transformer
  ↓
next-token prediction
```

The scale and architecture change considerably, but the underlying themes remain familiar: combine values using learned parameters, calculate an output, measure an error and adjust those parameters to reduce it.

## Related RC2014 Experiments

There is also a rather different exploration of artificial neurons in my [RC2014 project](https://github.com/davewalker5/RC2014), which contains programs written for the Z80-based RC2014 Mini II.

Implementing neuron concepts on an 8-bit retrocomputer is obviously a very different exercise from building them in Python, but the two projects share the same appeal: reducing the idea to something small enough that the individual operations remain visible.

## Getting Started

The [project Wiki](https://github.com/davewalker5/NeuralNetwork/wiki) contains the detailed documentation, background material and instructions for running and exploring the individual examples.

In particular, the Wiki is intended to document not just **how** the programs work, but **what each experiment demonstrates** and how each stage relates to the next.

## Authors

* **Dave Walker** - *Initial work*

## Feedback

To file issues or suggestions, please use the [Issues](https://github.com/davewalker5/NeuralNetwork/issues) page for this project on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
