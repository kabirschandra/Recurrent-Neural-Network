#ifndef RNN_H
#define RNN_H
#include <vector>
#include <cmath>
#include <iostream>
#include "matrix.h++"
#define ASSERT_VAR(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: (" << #condition << ")\n" \
                      << "Message: " << message << "\n" \
                      << "File: " << __FILE__ \
                      << ", Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)


class Layer {
    public:
    Matrix Wih; //Input weights 
    Matrix Ht; //Input to current layer (x)

    Matrix Whh; //Hidden weights
    Matrix Htprev; //Input (hidden)

    Matrix B; //Bias

    size_t dIn; //Input neurons
    size_t dOut; //Output neurons
};


class RNN {
    private:
    size_t nLayers;
    std::vector<Layer> layers; 
    Matrix yPred;

    public:
    /// Brief: Creates an RNN with the specified input, hidden, and output dimensions.
    /// @param layerDimensions Neuron count for each layer, including input and output.
    /// @return None.
    RNN(std::vector<size_t> layerDimensions) {
        ASSERT_VAR(layerDimensions.size() >= 2, "RNN cannot accept zero length layer dimension vector\n");

        this->layers.resize(layerDimensions.size() - 1);
        size_t nLayers = layerDimensions.size() - 1; 
        this->nLayers = nLayers;

        auto xavier_uniform = [](size_t fanIn, size_t fanOut) -> float {
            return sqrt(6.0f / (fanIn + fanOut));
        };

        for(size_t i = 0; i < nLayers; i++) {
            size_t dIn = layerDimensions[i];
            size_t dOut = layerDimensions[i + 1];
            Layer &currentLayer = this->layers[i];
            float b = xavier_uniform(dIn, dOut);

            currentLayer.dIn = dIn;
            currentLayer.dOut = dOut;

            currentLayer.Ht = Matrix(dIn, 0);
            currentLayer.Ht = currentLayer.Ht.fill(0.0f);

            currentLayer.Htprev = Matrix(dOut, 1);
            currentLayer.Htprev = currentLayer.Htprev.fill(0.0f);

            currentLayer.Wih = Matrix(dOut, dIn);
            currentLayer.Wih = currentLayer.Wih.uniform_initialisation(-b, b);

            currentLayer.Whh = Matrix(dOut, dOut);
            currentLayer.Whh = currentLayer.Whh.uniform_initialisation(-b, b);

            currentLayer.B = Matrix(dOut, 1);
            currentLayer.B = currentLayer.B.uniform_initialisation(-b, b);
        }
    }


    /// Brief: Prints each layer's weights, inputs, hidden states, and bias to standard output.
    /// @return None.
    void disp(void) {

        for(size_t i = 0; i < this->layers.size(); i++) {

            Layer &currentLayer = this->layers[i];

            std::cout << "======Layer: " << i << " =======\n";

            std::cout << "\t===== Wih =====" << "\n";
            currentLayer.Wih.disp();
            std::cout << "\n";

            std::cout << "\t===== Ht =====" << "\n";
            currentLayer.Ht.disp();
            std::cout << "\n";

            std::cout << "\t===== Whh=====" << "\n";
            currentLayer.Whh.disp();
            std::cout << "\n";

            std::cout << "\t===== HtPrev =====" << "\n";
            currentLayer.Htprev.disp();
            std::cout << "\n";

            std::cout << "\t===== B =====" << "\n";
            currentLayer.B.disp();

            std::cout << "\n\n\n";
        }


        return;
    }


    /// Brief: Runs the input sequence through the network and stores its predictions.
    /// @param x Input sequence with one input feature per row and one timestep per column.
    /// @return Predicted output sequence with one output neuron per row and one timestep per column.
    Matrix forward(Matrix &x) {
        ASSERT_VAR(x.get_rows() == this->layers.front().dIn, 
        "RNN forward pass dimension missmatch. Network contains "
         << this->layers.front().dIn << " input neurons, but x has " << x.get_rows() << " rows.\n");

        //Clear previous state
        for(Layer &layer : this->layers) {
            layer.Ht = Matrix(layer.dIn, 0);
            layer.Htprev = Matrix(layer.dOut, 1);
            layer.Htprev = layer.Htprev.fill(0.0f);
        }

        Matrix y = Matrix(this->layers.back().dOut, 0); //Will append each column to y
        for(size_t i = 0; i < x.get_cols(); i++) {
            //For each input on x

            this->layers.front().Ht = this->layers.front().Ht.append_cols(x.get_col(i));
            Matrix htC;
            for(size_t j = 0; j < this->nLayers; j++) {
                //For each layer in the network

                Layer &currentLayer = this->layers[j];

                htC = ((currentLayer.Wih * currentLayer.Ht.get_col(currentLayer.Ht.get_cols() - 1)) + 
                (currentLayer.Whh * currentLayer.Htprev.get_col(currentLayer.Htprev.get_cols() - 1)) + 
                currentLayer.B).activate(tanhf);

                //currentLayer.Htprev = htC;
                currentLayer.Htprev = currentLayer.Htprev.append_cols(htC);

                if(j != this->nLayers - 1) { //Append to next layer
                    Layer &nextLayer = this->layers[j + 1];
                    //nextLayer.Ht = htC;
                    nextLayer.Ht = nextLayer.Ht.append_cols(htC);

                } else { //Final layer, append to y
                    y = y.append_cols(htC);
                }
            }

        }

        this->yPred = y;
        return y;
    }

    /// Brief: Calculates gradients by backpropagation through time and updates network parameters.
    /// @param yActual Target sequence matching the most recent predictions.
    /// @param lr Positive learning rate used for the parameter updates.
    /// @return None.
    /// Requires a preceding non-empty forward pass with matching target dimensions.
    void backward(Matrix &yActual, float lr) {
        ASSERT_VAR(
            (this->yPred.get_rows() == yActual.get_rows()) &&
            (this->yPred.get_cols() == yActual.get_cols()),
            "Prediction matrix is: " << this->yPred.get_rows() << " x " << this->yPred.get_cols()
            << " but ground truth matrix is: "
            << yActual.get_rows() << " x " << yActual.get_cols() << "\n"
        );

        ASSERT_VAR(
            lr > 0.0f,
            "Learning rate must be greater than zero. Rate: " << lr << "\n"
        );

        ASSERT_VAR(
            this->yPred.get_cols() > 0,
            "Cannot perform backward pass without a non-empty forward pass."
        );

        //Temporal path (Shared component)
            auto delta_t = [this, &yActual](auto&& self, size_t n, size_t t) -> Matrix {
            //Should really cache deltas

            //Direct loss gradient: MSE gradient at the output layer
            Matrix gt(this->layers[n].dOut, 1);
            gt = gt.fill(0.0f);

            if(n == this->nLayers - 1) {
                Matrix error = this->yPred.get_col(t) - yActual.get_col(t);

                gt = error.scalar_multiply(2.0f / (float)(this->layers[n].dOut * yActual.get_cols()));
            }


            //Spatial gradient contribution: Wih^(n+1)^T * delta_t^(n+1)
            Matrix spatial(this->layers[n].dOut, 1);
            spatial = spatial.fill(0.0f);

            if (n + 1 < this->nLayers) {
                spatial = this->layers[n + 1].Wih.transpose() * self(self, n + 1, t);
            }


            //Temporal contribution: Whh^T * delta(t+1)
            Matrix temporal(this->layers[n].dOut, 1);
            temporal = temporal.fill(0.0f);

            if (t + 1 < yActual.get_cols()) {
                temporal = this->layers[n].Whh.transpose() * self(self, n, t + 1);
            }


            //Tanh contribution
            Matrix h = this->layers[n].Htprev.get_col(t + 1);
            Matrix ones(this->layers[n].dOut, 1);
            ones = ones.fill(1.0f);
            Matrix tanhDerivative = ones - h.hadamard(h);


            Matrix delta = (gt + spatial + temporal).hadamard(tanhDerivative);

            return delta;
        };


        std::vector<Matrix> WihNew(this->nLayers);
        std::vector<Matrix> WhhNew(this->nLayers);
        std::vector<Matrix> BNew(this->nLayers);
        for(size_t i = this->nLayers; i-- > 0;) { //For each layer

            Layer &currentLayer = this->layers[i];
            size_t tMax = yPred.get_cols();


            Matrix dWih(currentLayer.dOut, currentLayer.dIn);
            Matrix dWhh(currentLayer.dOut, currentLayer.dOut);
            Matrix dB(currentLayer.dOut, 1);
            dWih = dWih.fill(0.0f);
            dWhh = dWhh.fill(0.0f);
            dB = dB.fill(0.0f);
            for(size_t j = 0; j < tMax; j++) {


                Matrix delta = delta_t(delta_t, i, j);
                
                dWih = dWih + delta * currentLayer.Ht.get_col(j).transpose();
                dWhh = dWhh + delta * currentLayer.Htprev.get_col(j).transpose();
                dB = dB + delta;
            }

            WihNew[i] = (currentLayer.Wih - dWih.scalar_multiply(lr));
            WhhNew[i] = (currentLayer.Whh - dWhh.scalar_multiply(lr));
            BNew[i] = (currentLayer.B - dB.scalar_multiply(lr));
        }
       
        
        for(size_t i = this->nLayers; i-- > 0;) {
            
            Layer &currentLayer = this->layers[i];

            currentLayer.Wih = WihNew[i];
            currentLayer.Whh = WhhNew[i];
            currentLayer.B = BNew[i];
        }

        return;
    }



    /// Brief: Calculates mean squared error for the most recent forward-pass predictions.
    /// @param yActual Target sequence matching the stored predictions.
    /// @return Mean squared error averaged over output neurons and timesteps.
    float mse(const Matrix &yActual) const {
        ASSERT_VAR(
            this->yPred.get_rows() == yActual.get_rows() &&
            this->yPred.get_cols() == yActual.get_cols(),
            "RNN::mse(): Prediction and target dimensions do not match."
        );

        ASSERT_VAR(
            this->yPred.get_rows() > 0 && this->yPred.get_cols() > 0,
            "RNN::mse(): No non-empty forward-pass predictions available."
        );

        float sumSquaredError = 0.0f;

        for (size_t r = 0; r < this->yPred.get_rows(); r++) {
            for (size_t t = 0; t < this->yPred.get_cols(); t++) {
                float error = this->yPred.at(r, t) - yActual.at(r, t);
                sumSquaredError += error * error;
            }
        }

        return sumSquaredError /
            (float)(this->yPred.get_rows() * this->yPred.get_cols());
    }

};



#endif
