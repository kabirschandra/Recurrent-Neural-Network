#ifndef RNN_H
#define RNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include <random>
#include <thread>
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


        //Gradients
        matrix::Matrix<float> dWax;
        matrix::Matrix<float> dWaa;
        matrix::Matrix<float> dba;
    };

    class OutputLayer {
        public:
        //Output layer - yT = act(Wya @ aT + bY)
        matrix::Matrix<float> Wya; //Weight matrix of output
        matrix::Matrix<float> by; //Bias
        matrix::Matrix<float> y; //Output
        ACTIVATION_FUNCTION actO; //Activation function of output layer

        //Gradients
        matrix::Matrix<float> dWya;
        matrix::Matrix<float> dby;
    };

    class RNN {
        private:
        rnn::InputLayer inputLayer;
        std::vector<rnn::RecurrentLayer> recurrentLayers;
        rnn::OutputLayer outputLayer;

        public:

        /**
         * @brief Construct a recurrent neural network with the specified layer dimensions.
         *
         * @param layerDimensions Sizes of the input, recurrent, and output layers, in order.
         * @param actO Activation function configured for the output layer.
         */
        RNN(std::vector<size_t> layerDimensions, ACTIVATION_FUNCTION actO) {
            //Vector example: {3, 4, 3, 5}

            size_t nLayers = layerDimensions.size();
            assert(nLayers > 2);
            for(auto n : layerDimensions) {
                assert(n > 0);
            }


            //Handle input layer
            size_t nFirst = layerDimensions.front();
            this->inputLayer.x.redimension(nFirst, 1);


            this->recurrentLayers.resize(nLayers - 2);
            //Handle recurrent layer
            for(size_t i = 0; i < nLayers - 2; i++) {
                size_t nX = layerDimensions[i];
                size_t nA = layerDimensions[i + 1];

                RecurrentLayer *current = &(this->recurrentLayers[i]);
                current->A.redimension(0, 0);
                current->Wax.redimension(nA, nX);
                current->Wax.xavier_uniform();
                current->Waa.redimension(nA, nA);
                current->Waa.xavier_uniform();

                current->ba.redimension(nA, 1);
                current->ba.fill(0); //Zero initialisation

                //Gradients
                current->dWax.redimension(nA, nX);
                current->dWax.fill(0);

                current->dWaa.redimension(nA, nA);
                current->dWaa.fill(0);

                current->dba.redimension(nA, 1);
                current->dba.fill(0);
            }

            //Handle output layer
            size_t nLast = layerDimensions.back();
            size_t nLastPrev = layerDimensions[layerDimensions.size() - 2];
            this->outputLayer.y.redimension(nLast, 1);

            this->outputLayer.by.redimension(nLast, 1);
            this->outputLayer.by.fill(0);

            this->outputLayer.Wya.redimension(nLast, nLastPrev);
            this->outputLayer.Wya.xavier_uniform();
            this->outputLayer.actO = actO;

            //Gradients
            this->outputLayer.dWya.redimension(nLast, nLastPrev);
            this->outputLayer.dWya.fill(0);

            this->outputLayer.dby.redimension(nLast, 1);
            this->outputLayer.dby.fill(0);
        }


        /**
         * @brief Process one input time step and store the resulting recurrent states and output.
         *
         * @param x Input column vector for the current time step.
         * @return void
         */
        void forward(matrix::Matrix<float> &x) { // STEP forward (column vector)

            this->inputLayer.x.append_column(x);

            matrix::Matrix<float> input = x;

            for(auto &currentLayer : this->recurrentLayers) {

                size_t nA = currentLayer.Waa.get_rows();

                // Previous hidden state of THIS recurrent layer
                matrix::Matrix<float> aPrevious(nA, 1);

                if(currentLayer.A.get_cols() == 0) {
                    aPrevious.fill(0);
                }
                else {
                    aPrevious = currentLayer.A.get_column(currentLayer.A.get_cols() - 1);
                }

                // Calculate the new hidden state
                matrix::Matrix<float> inputContribution = currentLayer.Wax * input;
                matrix::Matrix<float> recurrentContribution = currentLayer.Waa * aPrevious;

                matrix::Matrix<float> aT = inputContribution
                                        + recurrentContribution
                                        + currentLayer.ba;

                aT.activate_tanh();

                // Save hidden state for BPTT
                currentLayer.A.append_column(aT);

                // This layer's output becomes the next layer's input
                input = aT;
            }

            // Calculate network output from final recurrent layer
            matrix::Matrix<float> yT = this->outputLayer.Wya * input
                                    + this->outputLayer.by;

            // Apply output activation according to outputLayer.actO
            // (TANH, SIGMOID, RELU, etc.)

            this->outputLayer.y = yT;

            return;
        }


        /**
         * @brief Clear the stored recurrent states.
         *
         * @return void
         */
        void reset(void) {
            for(auto &currentLayer : this->recurrentLayers) {
                currentLayer.A.redimension(0, 0); //Clear internal state
            }
            return;
        }


        /**
         * @brief Perform the backward pass.
         *
         * @return void
         */
        void backward() {
            return;
        }

    };

};


#endif
