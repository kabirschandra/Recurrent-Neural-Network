#ifndef RNN_H
#define RNN_H
#include <vector>
#include <cmath>
#include <iostream>
#include "matrix.h++"



class Layer {
    public:
    Matrix Wih; //Input weights 
    Matrix Xt; //Input (x)

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

        this->layers.resize(layerDimensions.size() - 1);
        size_t nLayers = layerDimensions.size(); 
        this->nLayers = nLayers;

        auto xavier_uniform = [](size_t fanIn, size_t fanOut) -> float {
            return sqrt(6.0f / (fanIn + fanOut));
        };

        for(size_t i = 0; i < nLayers - 1; i++) {
            size_t dIn = layerDimensions[i];
            size_t dOut = layerDimensions[i + 1];
            Layer &currentLayer = this->layers[i];
            float b = xavier_uniform(dIn, dOut);

            currentLayer.dIn = dIn;
            currentLayer.dOut = dOut;

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

            std::cout << "\t===== Xt =====" << "\n";
            currentLayer.Xt.disp();
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
        Matrix y = Matrix(this->layers.back().dOut, 0);
        
        for(size_t i = 0; i < x.get_cols(); i++) {
            this->layers.front().Xt = x.get_col(i);


            for(size_t j = 0; j < this->nLayers; j++) {
                Layer &currentLayer = this->layers[j];


                if(i == 0) { //Must allocate h(t-1)

                    auto xavier_uniform = [](size_t fanIn, size_t fanOut) -> float {
                        return sqrt(6.0f / (fanIn + fanOut));
                    };
                    float b = xavier_uniform(currentLayer.dIn, currentLayer.dOut);
                    currentLayer.Htprev = Matrix(x.get_cols(), 1);
                    currentLayer.Htprev.uniform_initialisation(-b, b);
                }


                if(j != nLayers - 1) {
                    Layer &nextLayer = this->layers[j + 1];
                    nextLayer.Xt = (((currentLayer.Wih * currentLayer.Xt) + (currentLayer.Whh * currentLayer.Htprev)) + currentLayer.B).activate(tanhf);


                } else { //Last layer outputs to y matrix
                    Matrix yi = (((currentLayer.Wih * currentLayer.Xt) + (currentLayer.Whh * currentLayer.Htprev)) + currentLayer.B).activate(tanhf); //Col vector
                    y = y.append_cols(yi);
                }
            }

        }

        return y;
    }

};



#endif
