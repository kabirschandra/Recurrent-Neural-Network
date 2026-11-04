#include "./src/matrix.h++"
#include "./src/rnn.h++"

int main(void) {


    RNN rnn = RNN({3,4,2});
    rnn.disp();


    Matrix f = Matrix(3, 4, {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    });
    Matrix y = rnn.forward(f);


    return 0;
}

