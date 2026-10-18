#ifndef RNN_H
#define RNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include "matrix.h++"




namespace rnn {
    
    typedef enum ACTIVATION_FUNCTION {
        RELU,
        SIGMOID,
        TANH, 
    } ACTIVATION_FUNCTION;

    class Layer {
        public:

        //Hidden state - aT = tanh(Wax @ xT + Waa @ aT_1 + bA)
        matrix::Matrix<float> A; //Hidden state
        matrix::Matrix<float> Wax; //Weight matrix of input
        matrix::Matrix<float> x; //Input
        matrix::Matrix<float> Waa; //Weight matrix of hidden state
        matrix::Matrix<float> prevA; //Previous hidden state
        matrix::Matrix<float> ba; //Bias
        ACTIVATION_FUNCTION actH; //Activation function of hidden state

        //Output layer - yT = act(Wya @ aT + bY)
        matrix::Matrix<float> Wya; //Weight matrix of output
        matrix::Matrix<float> by; //Bias
        ACTIVATION_FUNCTION actO; //Activation function of output layer
    };


    class RNN {
        private:
        std::vector<rnn::Layer> layers;

        public:

        RNN(std::vector<size_t> layerDimensions) {

            size_t nLayers = layerDimensions.size();
            layers.resize(nLayers);

            for(size_t i = 0; i < nLayers; i++) {
                rnn::Layer &currentLayer = this->layers[i];                
                
                
            }

        }


    };

};


#endif

