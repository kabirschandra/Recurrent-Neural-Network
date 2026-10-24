#ifndef RNN_H
#define RNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>
#include "matrix.h++"


//TODO: Xavier uniform on Wax, Waa, Wya



namespace rnn {
    
    typedef enum ACTIVATION_FUNCTION {
        RELU,
        SIGMOID,
        TANH, 
    } ACTIVATION_FUNCTION;

    class InputLayer {
        public:
        //Input layer to network
        matrix::Matrix<float> x;
    };

    class RecurrentLayer {
        public:
        //Hidden state - aT = act(Wax @ xT + Waa @ aT_1 + ba)
        matrix::Matrix<float> A; //Hidden states across time
        matrix::Matrix<float> Wax; //Input weights
        matrix::Matrix<float> Waa; //Recurrent weights
        matrix::Matrix<float> ba;  //Bias
        ACTIVATION_FUNCTION actH = TANH; //Hidden activation function (Default)
    };

    class OutputLayer {
        public:
        //Output layer - yT = act(Wya @ aT + bY)
        matrix::Matrix<float> Wya; //Weight matrix of output
        matrix::Matrix<float> by; //Bias
        matrix::Matrix<float> y; //Output
        ACTIVATION_FUNCTION actO; //Activation function of output layer
    };

    class RNN {
        private:
        rnn::InputLayer inputLayer;
        std::vector<rnn::RecurrentLayer> layers;
        rnn::OutputLayer outputLayer;

        public:

        RNN(std::vector<size_t> layerDimensions, ACTIVATION_FUNCTION actO) {
            //Vector example: {3, 4, 3, 5}

            size_t nLayers = layerDimensions.size();
            assert(nLayers > 2);


            //Handle input layer
            size_t nFirst = layerDimensions.front();
            this->inputLayer.x.redimension(nFirst, 1);


            this->layers.resize(nLayers - 2);
            //Handle recurrent layer
            for(size_t i = 0; i < nLayers - 2; i++) {
                size_t nX = layerDimensions[i];
                size_t nA = layerDimensions[i + 1];

                RecurrentLayer *current = &(this->layers[i]);
                current->Wax.redimension(nA, nX);
                current->Waa.redimension(nA, nA);


                current->ba.redimension(nA, 1);
                current->ba.fill(0); //Zero initialisation
            }

            //Handle output layer
            size_t nLast = layerDimensions.back();
            size_t nLastPrev = layerDimensions[layerDimensions.size() - 2];
            this->outputLayer.y.redimension(nLast, 1);

            this->outputLayer.by.redimension(nLast, 1);
            this->outputLayer.by.fill(0);

            this->outputLayer.Wya.redimension(nLast, nLastPrev);
            this->outputLayer.actO = actO;

        }


    };

};


#endif

