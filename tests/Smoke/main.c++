#include <cmath>
#include <iostream>
#include "../../src/rnn.h++"

int main() {
    RNN rnn({1, 4, 1});
    Matrix x(1, 3);
    Matrix target(1, 3);
    for (std::size_t t = 0; t < 3; ++t) {
        x.at(0, t) = std::sin(static_cast<float>(t) * 0.1f);
        target.at(0, t) = std::sin(static_cast<float>(t + 1) * 0.1f);
    }
    Matrix prediction = rnn.forward(x);
    if (prediction.get_rows() != 1 || prediction.get_cols() != 3) return 1;
    float manual = 0.0f;
    for (std::size_t t = 0; t < 3; ++t) {
        float d = prediction.at(0, t) - target.at(0, t);
        manual += d * d;
    }
    manual /= 3.0f;
    if (std::fabs(manual - rnn.mse(target)) > 1e-6f) {
        std::cerr << "MSE method does not match independently computed MSE\n";
        return 1;
    }
    rnn.backward(target, 0.00001f);
    rnn.forward(x);
    if (!std::isfinite(rnn.mse(target))) return 1;
    std::cout << "Smoke test passed: forward, mse, backward\n";
}
