#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "../../src/rnn.h++"

int main(int argc, char** argv) {
    const std::size_t epochs = argc > 1 ? std::stoull(argv[1]) : 10000;
    const float lr = argc > 2 ? std::stof(argv[2]) : 0.0001f;
    const std::size_t T = argc > 3 ? std::stoull(argv[3]) : 5;
    const float step = argc > 4 ? std::stof(argv[4]) : 0.1f;
    if (!epochs || !T || !(lr > 0) || !(step > 0)) { std::cerr << "Invalid arguments\n"; return 1; }
    RNN rnn({1, 4, 1});
    Matrix x(1, T), target(1, T);
    for (std::size_t t = 0; t < T; ++t) {
        x.at(0, t) = std::sin(static_cast<float>(t) * step);
        target.at(0, t) = std::sin(static_cast<float>(t + 1) * step);
    }
    std::ofstream history("tests/Sine/loss.csv");
    if (!history) { std::cerr << "Run from repository root\n"; return 1; }
    history << "epoch,mse\n";
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix prediction = rnn.forward(x);
        const float mse = rnn.mse(target);
        if (!std::isfinite(mse)) {
            std::cerr << "Non-finite MSE at epoch " << epoch << '\n';
            return 1;
        }
        history << epoch << ',' << std::setprecision(9) << mse << '\n';
        if (epoch % 100 == 0 || epoch + 1 == epochs) std::cout << "Epoch " << epoch << " | MSE: " << mse << '\n';
        rnn.backward(target, lr);
    }
    Matrix prediction = rnn.forward(x);
    std::cout << "Final MSE after training: " << rnn.mse(target) << '\n';
    std::ofstream predictions("tests/Sine/predictions.csv");
    if (!predictions) { std::cerr << "Cannot write predictions.csv\n"; return 1; }
    predictions << "time,input,target,prediction\n";
    for (std::size_t t = 0; t < T; ++t) {
        predictions << std::setprecision(9) << (t + 1) * step << ',' << x.at(0,t) << ',' << target.at(0,t) << ',' << prediction.at(0,t) << '\n';
    }
    std::cout << "Saved tests/Sine/loss.csv and tests/Sine/predictions.csv\n";
}
