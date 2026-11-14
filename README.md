# C++ based recurrent neural network

A single-threaded, multilayer recurrent neural network (RNN) written in C++. Uses only the STL. Implements a vanilla RNN with Tanh activation and Backpropagation Through Time (BPTT) for training on sequential data.

## Features

- CPU only

- Configurable number of recurrent layers and neurons per layer

- Tanh activation for hidden and output layers

- Many-to-many sequence processing

- Backpropagation Through Time (BPTT)

- Mean Squared Error (MSE) loss function

- Xavier uniform weight initialisation

- Custom matrix library for linear algebra operations

- Gradient descent weight updates

- Processes one sequence at a time

## Project Structure

| Directory | What's inside |
| ----- | ----- |
| `./src` | RNN and matrix source code |
| `./tests/Sine` | Sine-wave prediction test and Python plotting script |
| `./tests/Smoke` | Basic forward, backward and MSE validation |
| `./bin` | Compiled test binaries |
| `./run_tests.sh` | Test compilation, execution and cleanup script |

## Requirements

- C++17 or newer
- Linux or WSL
- No external C++ libraries
- Python 3 with Matplotlib for plotting test results

## Basic Usage

```cpp
#include <iostream>
#include <vector>
#include "./src/matrix.h++"
#include "./src/rnn.h++"

int main(void) {

    // 1 input neuron, 4 hidden neurons, 1 output neuron
    std::vector<size_t> layers = {1, 4, 1};

    // Create recurrent neural network
    RNN net(layers);

    // Create input sequence containing 5 timesteps
    Matrix x(1, 5);

    x.at(0, 0) = 0.0f;
    x.at(0, 1) = 0.1f;
    x.at(0, 2) = 0.2f;
    x.at(0, 3) = 0.3f;
    x.at(0, 4) = 0.4f;

    // Expected output sequence
    Matrix expected_output(1, 5);

    expected_output.at(0, 0) = 0.1f;
    expected_output.at(0, 1) = 0.2f;
    expected_output.at(0, 2) = 0.3f;
    expected_output.at(0, 3) = 0.4f;
    expected_output.at(0, 4) = 0.5f;

    // Perform forward pass
    Matrix output = net.forward(x);

    // Calculate mean squared error
    float loss = net.mse(expected_output);

    std::cout << "MSE: " << loss << "\n";

    // Perform backpropagation through time
    net.backward(expected_output, 0.001f);

    // Display network weights, biases and hidden states
    net.disp();

    return 0;
}
```

## Architecture

The network consists of one or more recurrent layers. Each layer maintains its own input weights, recurrent weights, biases and hidden states.

At each timestep, the hidden state is calculated as:

\[
h_t^{(l)} = \tanh\left(W_{ih}^{(l)}x_t^{(l)} + W_{hh}^{(l)}h_{t-1}^{(l)} + b^{(l)}\right)
\]

Where:

- \(W_{ih}\) is the input weight matrix
- \(W_{hh}\) is the recurrent weight matrix
- \(b\) is the bias vector
- \(x_t\) is the input to the layer at timestep \(t\)
- \(h_{t-1}\) is the previous hidden state

The output of each layer becomes the input to the next layer at the same timestep.

The final layer's hidden state is used directly as the network output.

### Input and Output

Inputs and outputs use the custom `Matrix` class.

Each column represents one timestep, while each row represents one neuron.

For example, a sequence with 3 input features and 5 timesteps has dimensions:

```text
Input matrix: 3 x 5

        t0  t1  t2  t3  t4
       --------------------
Input 1 | x   x   x   x   x
Input 2 | x   x   x   x   x
Input 3 | x   x   x   x   x
```

A network with 2 output neurons returns a `2 x 5` matrix.

The network processes all timesteps sequentially.

## Training

Training uses Backpropagation Through Time (BPTT).

For each training iteration:

1. Perform a forward pass over the input sequence.
2. Calculate the Mean Squared Error against the target sequence.
3. Calculate output gradients.
4. Propagate gradients backwards through the recurrent connections and preceding layers.
5. Accumulate weight and bias gradients across all timesteps.
6. Update weights and biases using gradient descent.

The network uses the Tanh derivative:

\[
\tanh'(z)=1-\tanh^2(z)
\]

Parameters are updated using:

\[
W_{\text{new}}=W_{\text{old}}-\eta\frac{\partial L}{\partial W}
\]

Where \(\eta\) is the learning rate.

### Mean Squared Error

The network uses Mean Squared Error as its loss function:

\[
MSE=\frac{1}{NT}\sum_{t=0}^{T-1}\sum_{i=0}^{N-1}(\hat{y}_{i,t}-y_{i,t})^2
\]

Where:

- \(N\) is the number of output neurons
- \(T\) is the number of timesteps
- \(\hat{y}\) is the predicted output
- \(y\) is the expected output

The `mse()` method calculates loss using the predictions stored from the most recent forward pass.

## Memory

The network allocates its weight matrices, recurrent weight matrices and biases during construction.

Hidden states and layer inputs are stored for every timestep during the forward pass. These values are required for Backpropagation Through Time.

Gradient matrices are allocated during the backward pass and accumulated across the sequence.

The network resets its hidden states to zero at the beginning of each forward pass.

Memory usage increases with sequence length, number of layers and number of neurons.

Unlike a pre-allocated network, the current implementation dynamically grows its hidden-state matrices as additional timesteps are processed.

## Limitations

- Batch size fixed at 1 sequence

- No GPU support

- Single-threaded only

- Tanh activation only

- No LSTM or GRU support

- No gradient clipping

- No truncated BPTT

- No built-in optimiser beyond gradient descent

- No model save/load functionality

- Hidden states reset between forward passes

- Output values are limited to the Tanh range of -1 to 1

- Recursive BPTT implementation recalculates intermediate gradients rather than caching them

## Testing

The library includes tests for basic network functionality and sine-wave sequence prediction.

The sine test trains the RNN to predict the next value in a sine-wave sequence.

The Python plotting script generates graphs of the training loss and predicted outputs.

### Run All Tests

```bash
chmod +x run_tests.sh
./run_tests.sh all
```

### Individual Tests

| Test | Location | What it tests |
|------|----------|----------------|
| Smoke | `tests/Smoke/main.c++` | Forward pass, MSE calculation and backward pass |
| Sine | `tests/Sine/main.c++` | Sequential learning and BPTT |
| Sine Plot | `tests/Sine/plot_sine.py` | Training loss and prediction visualisation |

### Running Tests

Use the test runner script:

```bash
# Display available commands
./run_tests.sh help

# Compile tests
./run_tests.sh compile

# Run smoke test
./run_tests.sh smoke

# Run sine-wave training
./run_tests.sh run

# Generate plots
./run_tests.sh plot

# Compile, run and plot
./run_tests.sh all

# Clean generated files
./run_tests.sh clean
```

### Custom Training Parameters

The sine test accepts configurable training parameters:

```bash
./run_tests.sh all 10000 0.0001 5 0.1
```

Arguments:

| Argument | Description |
|----------|-------------|
| `10000` | Number of training epochs |
| `0.0001` | Learning rate |
| `5` | Number of timesteps |
| `0.1` | Step size between sine samples |

### Plotting Results

Install Matplotlib:

```bash
python3 -m pip install matplotlib
```

After running the sine test, the plotting script produces:

- `loss.csv` — MSE recorded during training
- `predictions.csv` — Target and predicted values
- `loss.png` — Training loss against epoch
- `predictions.png` — Predicted sine wave against the target

These files are generated inside `tests/Sine/`.

## Future Development

Potential improvements include:

- Gradient clipping for more stable training
- Truncated Backpropagation Through Time
- Additional activation functions
- Mini-batch training
- Adam and momentum-based optimisers
- Model save/load functionality
- Configurable initial hidden states
- LSTM and GRU implementations
- Numerical gradient checking for BPTT validation
