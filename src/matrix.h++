#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <iostream>
#include <random>

class Matrix {
    private:
    size_t rows;
    size_t cols;
    std::vector<float> data;

    public:
    Matrix() {
        this->rows = 0;
        this->cols = 0;
    }
    Matrix(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols);
    }
    Matrix(size_t rows, size_t cols, std::vector<float> data) : Matrix(rows, cols) {

        for(size_t i = 0; i < this->data.size(); i++) {
            this->data[i] = data[i];
        }
    }
    void disp(void) {
        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols; c++) {
                std::cout << this->at(r, c) << " ";
            }
            std::cout << "\n";
        }

        return;
    }

    size_t get_rows(void) {
        return this->rows;
    }
    size_t get_cols(void) {
        return this->cols;
    }
    float &at(size_t row, size_t col) {
        size_t nCols = this->cols;
        size_t nRows = this->rows;
        return this->data[(row * nCols) + col];
    }
    void resize(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols);
        return;
    }
    void fill(float v) {
        for(auto &i : this->data) {
            i = v;
        }
        return;
    }
    void uniform_initialisation(float min, float max) {

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        for(auto &i : this->data) {
            i = dist(gen);
        }
        return;
    }
    Matrix get_col(size_t col) {
        Matrix colM(this->rows, 1);
        for(size_t i = 0; i < this->rows; i++) {
            colM.at(i, 0) = this->at(i, col);
        }
        return colM;
    }
    Matrix get_row(size_t row) {
        Matrix rowM(1, this->cols);
        for(size_t i = 0; i < this->rows; i++) {
            rowM.at(0, i) = this->at(row, i);
        }
        return rowM;
    }
    Matrix append_rows(Matrix &m) {
        Matrix result = *this;

        result.rows += m.rows;
        result.data.reserve(this->data.size() + m.data.size());
        result.data.insert(this->data.end(), m.data.begin(), m.data.end());
        return result;
    }
    Matrix append_cols(Matrix &m) {
        Matrix result(this->rows, this->cols + m.cols);

        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols + m.cols; c++) {
                if(c > this->cols - 1) {
                    result.at(r, c) = m.at(r, c - this->cols);
                } else {
                    result.at(r, c) = this->at(r, c);
                }
            }
        }

        return result;
    }

    Matrix operator+(Matrix &rhs) {
        Matrix result(this->rows, this->cols);
        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] + rhs.data[i];
        }
        return result;
    }
    Matrix operator-(Matrix &rhs) {
        Matrix result(this->rows, this->cols);
        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] - rhs.data[i];
        }
        return result;
    }
    Matrix operator*(Matrix &rhs) {
        Matrix result(this->rows, rhs.cols);
        for(size_t lhsr = 0; lhsr < this->rows; lhsr++) {
            for(size_t rhsc = 0; rhsc < rhs.cols; rhsc++) {

                float acc = 0;
                for(size_t lhsc = 0; lhsc < this->cols; lhsc++) {
                    acc += this->at(lhsr, lhsc) * rhs.at(lhsc, rhsc);
                }
                result.at(lhsr, rhsc) = acc;
            }
        }

        return result;
    }
    Matrix hadamard(Matrix &rhs) {
        Matrix result(this->rows, this->cols);
        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] * rhs.data[i];
        }
        return result;
    }
    Matrix scalar_multiply(float scalar) {
        Matrix result(this->rows, this->cols);
        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }
    Matrix transpose(void) {
        Matrix result(this->cols, this->rows);
        
        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols; c++) {
                result.at(c, r) = this->at(r, c);
            }
        }

        return result;
    }

};




#endif
