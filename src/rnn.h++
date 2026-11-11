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
    Matrix yPred;

    public:
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

    void backward(Matrix &yActual) {
        ASSERT_VAR((this->yPred.get_rows() == yActual.get_rows()) 
        && (this->yPred.get_cols() == yActual.get_cols()), 
        "RNN backward pass input mismatch. Prediction matrix is: " 
        << yActual.get_rows() << " x " << yActual.get_cols() 
        << " but ground truth matrix is: " 
        << yPred.get_rows() << " x " << yPred.get_cols() << ".\n");


        auto mse = [](Matrix yPred, Matrix yActual, size_t col) -> float {
            float loss = 0.0f; 

            Matrix yPredCurrent = yPred.get_col(col);
            Matrix yActualCurrent = yActual.get_col(col);

            for(size_t i = 0; i < yPredCurrent.get_rows(); i++) {
                loss += powf((yPredCurrent.at(i, 1) - yActualCurrent.at(i, 1)), 2);
            }

            loss /= (float)(yPredCurrent.get_rows());

            return loss;
        };


        for(size_t i = this->nLayers; i != SIZE_MAX; i--) { //For each layer

            //Temporal path (Shared component)
            

            //Wih


            //Whh


            //B


        }
        
        return;
    }

};



#endif
