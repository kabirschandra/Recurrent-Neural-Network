#include "./src/matrix.h++"
#include "./src/rnn.h++"


int main(void) {

    rnn::RNN rnn = rnn::RNN({3,4,5,5,4,3}, rnn::ACTIVATION_FUNCTION::TANH);

    return 0;
}

