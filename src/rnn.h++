#ifndef RNN_H
#define RNN_H
#include <vector>
#include <cmath>
#include <iostream>
#include "matrix.h++"
// Custom assertion with descriptive error messages.
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


    public:
    RNN(std::vector<size_t> layerDimensions) {
        ASSERT_VAR(layerDimensions.size() >= 2, "RNN cannot accept zero length layer dimension vector");

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

            currentLayer.Ht = Matrix(dIn, 1);
            currentLayer.Ht.uniform_initialisation(-b, b);

            currentLayer.Htprev = Matrix(dOut, 1);
            currentLayer.Htprev.uniform_initialisation(-b, b);

            currentLayer.Wih = Matrix(dOut, dIn);
            currentLayer.Wih.uniform_initialisation(-b, b);

            currentLayer.Whh = Matrix(dOut, dOut);
            currentLayer.Whh.uniform_initialisation(-b, b);

            currentLayer.B = Matrix(dOut, 1);
            currentLayer.B.uniform_initialisation(-b, b);
        }
    }


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


    Matrix forward(Matrix &x) {
        ASSERT_VAR(x.get_rows() == this->layers.front().dIn, 
        "RNN forward pass dimension missmatch. Network contains "
         << this->layers.front().dIn << " input neurons, but x has " << x.get_rows() << " rows.");

        Matrix y = Matrix(this->layers.back().dOut, 0); //Will append each column to y
        for(size_t i = 0; i < x.get_cols(); i++) {
            //For each input on x

            this->layers.front().Ht = x.get_col(i);
            Matrix htC;
            for(size_t j = 0; j < this->nLayers; j++) {
                //For each layer in the network

                Layer &currentLayer = this->layers[j];

                htC = ((currentLayer.Wih * currentLayer.Ht) + 
                (currentLayer.Whh * currentLayer.Htprev) + 
                currentLayer.B).activate(tanhf);

                currentLayer.Htprev = htC;


                if(j != this->nLayers - 1) { //Append to next layer
                    Layer &nextLayer = this->layers[j + 1];
                    nextLayer.Ht = htC;

                } else { //Final layer, append to y
                    y = y.append_cols(htC);
                }
            }

        }


        return y;
    }

};



#endif
