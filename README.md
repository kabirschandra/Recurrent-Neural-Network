# C++ based recurrent neural network

A single-threaded, multilayer vanilla recurrent neural network written in C++. Uses only the STL and a custom matrix class. Built to understand how recurrent networks work, particularly hidden states and backpropagation through time (BPTT), rather than to be fast or production-ready.

## Features

- CPU only
- Configurable layer sizes
- Tanh activation in every recurrent layer
- Many-to-many sequence processing
- Backpropagation through time (BPTT)
- Mean squared error (MSE) loss
- Xavier uniform initialisation
- Gradient descent updates
- Custom matrix operations; no external C++ libraries

## Project Structure

| Directory | What's inside |
| --- | --- |
| `./src` | RNN and matrix headers |
| `./tests/Smoke` | Basic forward, loss and backward checks |
| `./tests/Sine` | Sine-wave prediction and plotting |
| `./tests/XOR` | Two-step XOR experiment |
| `./tests/MNIST` | Handwritten-digit sequence experiment |
| `./build` | Compiled test binaries |
| `./run_tests.sh` | Test runner |

## Requirements

- C++17 or newer (`g++` or `clang++`)
- Linux or WSL
- Python 3 and Matplotlib for the sine plots only
- MNIST IDX dataset files for the MNIST test

## Basic Usage

```cpp
#include <iostream>
#include <vector>
#include "src/rnn.h++"

int main() {
    // 1 input, 4 hidden neurons, 1 output
    RNN net({1, 4, 1});

    // Each column is one timestep
    Matrix input(1, 5);
    Matrix target(1, 5);

    for (size_t t = 0; t < 5; ++t) {
        input.at(0, t) = static_cast<float>(t) * 0.1f;
        target.at(0, t) = static_cast<float>(t + 1) * 0.1f;
    }

    // Forward pass, loss, then BPTT update
    Matrix output = net.forward(input);
    std::cout << "MSE: " << net.mse(target) << '\n';
    net.backward(target, 0.001f);

    return 0;
}
```

## Architecture

Each recurrent layer has input weights (`Wih`), recurrent weights (`Whh`), biases (`B`) and hidden states. At timestep `t`, layer `l` calculates:

$$
h_t^{(l)} = \tanh\left(W_{ih}^{(l)}x_t^{(l)} + W_{hh}^{(l)}h_{t-1}^{(l)} + b^{(l)}\right)
$$

- `Wih` maps the current input into the hidden layer.
- `Whh` maps the previous hidden state into the current hidden state.
- `B` is the bias vector.
- `x` is the current input (or the previous layer's output).
- `h` is the hidden state.

The final layer's hidden state is the network output. The input matrix has dimensions `features x timesteps`; the output matrix has dimensions `output neurons x timesteps`. Hidden states start at zero for each forward pass.

## Training

Training uses full backpropagation through time. The forward pass stores hidden states for each timestep. The backward pass propagates error through the layers and backward through time, accumulates gradients and updates the weights and biases.

The Tanh derivative is:

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
$$

Gradient descent updates each parameter using:

$$
W_{\mathrm{new}} = W_{\mathrm{old}} - \eta\frac{\partial L}{\partial W}
$$

Here, `eta` is the learning rate.

### Mean Squared Error

The network uses MSE across all output neurons and timesteps:

$$
\mathrm{MSE} = \frac{1}{NT}\sum_{t=0}^{T-1}\sum_{i=0}^{N-1}(\hat{y}_{i,t} - y_{i,t})^2
$$

- `N` is the number of output neurons.
- `T` is the number of timesteps.
- `y_hat` is the predicted output.
- `y` is the target output.

Call `mse(target)` after `forward(input)`. The backward-pass output gradient should use the same `1 / (N * T)` normalisation for consistency.

## Memory

Weights and biases are created when the network is constructed. The forward pass stores activations across timesteps so BPTT can use them later. Hidden-state matrices grow with sequence length, and the backward pass creates gradient matrices.

This implementation favours code that is easy to follow over minimising allocations or memory use.

## Limitations

- Processes one sequence at a time; no mini-batches
- Single-threaded CPU implementation
- Tanh activation only
- No LSTM or GRU gates
- No gradient clipping or truncated BPTT
- Basic gradient descent only
- Hidden state resets between forward passes
- Recursive BPTT recalculates intermediate gradients and becomes very slow for longer sequences
- No save/load support

These are deliberate trade-offs for a learning project. The aim is to implement the forward pass, state handling and gradients myself, not to compete with established ML frameworks.

## Testing

The test runner is in the repository root:

```bash
chmod +x run_tests.sh
./run_tests.sh help
```

| Test | Location | What it tests |
| --- | --- | --- |
| Smoke | `tests/Smoke/main.c++` | Forward pass, MSE and backward-pass sanity checks |
| Sine | `tests/Sine/main.c++` | Next-step prediction on a sine sequence |
| XOR | `tests/XOR/main.c++` | Learning XOR from a two-timestep input sequence |
| MNIST | `tests/MNIST/main.c++` | Treating image rows as a sequence for digit classification |

### Commands

```bash
# Compile all tests
./run_tests.sh compile

# Run basic checks
./run_tests.sh smoke

# Sine: epochs, learning rate, timesteps, step size
./run_tests.sh run 5000 0.001 32 0.2

# Plot sine results (requires matplotlib)
./run_tests.sh plot

# XOR: epochs, learning rate
./run_tests.sh xor 10000 0.01

# MNIST: epochs, training images, test images, learning rate
./run_tests.sh mnist 1 100 100 0.001

# Compile, smoke test, train sine and plot
./run_tests.sh all 5000 0.001 32 0.2

# Run all tests, including XOR and MNIST
./run_tests.sh all-extended

# Remove generated binaries and sine results
./run_tests.sh clean
```

Set `CXX=clang++` before a command to use Clang instead of the default `g++`.

### Sine plots

Install Matplotlib if needed:

```bash
python3 -m pip install matplotlib
```

The sine test writes `tests/Sine/loss.csv` and `tests/Sine/predictions.csv`. The plotting script produces `loss.png` and `predictions.png` in the same directory. The plotted predictions are on the training sequence, not a separate validation set.

### XOR

Each example contains two binary inputs over two timesteps. The test uses the first input bit as the target at the first timestep and XOR as the target at the second timestep. It checks all four combinations after training. Results can vary with random initialisation.

### MNIST

Download and decompress the four original MNIST IDX files and put them in `tests/MNIST/data/`:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

The MNIST test treats each 28x28 image as 28 timesteps with 28 features each. It uses 10 Tanh outputs and MSE rather than softmax/cross-entropy. Accuracy is measured from the final timestep.

**Note:** MNIST is included as an architecture experiment. With the current recursive BPTT implementation, even small runs may take a long time. High classification accuracy is not the goal of this project.
