
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>

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

class Matrix {
    private:
    size_t rows;
    size_t cols;
    std::vector<float> data;

    public:

    /// Brief: Creates an empty matrix with zero rows and columns.
    /// @return None.
    Matrix() {
        this->rows = 0;
        this->cols = 0;
    }

    /// Brief: Creates a matrix with a specific number of rows and columns.
    /// @param rows Number of rows in the matrix.
    /// @param cols Number of columns in the matrix.
    /// @return None.
    Matrix(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols);
    }

    /// Brief: Creates a matrix and copies values from a flat data vector into it.
    /// @param rows Number of rows in the matrix.
    /// @param cols Number of columns in the matrix.
    /// @param data Flat vector of values to initialise the matrix with.
    /// @return None.
    Matrix(size_t rows, size_t cols, std::vector<float> data) : Matrix(rows, cols) {
        ASSERT_VAR(data.size() == rows * cols,
            "Matrix constructor: Expected " << rows * cols
            << " elements, received " << data.size() << ".");

        for(size_t i = 0; i < this->data.size(); i++) {
            this->data[i] = data[i];
        }
    }

    /// Brief: Prints the matrix contents to standard output.
    /// @return None.
    void disp(void) const {
        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols; c++) {
                std::cout << this->at(r, c) << " ";
            }
            std::cout << "\n";
        }

        return;
    }

    /// Brief: Returns the current number of rows.
    /// @return Number of rows.
    size_t get_rows(void) const {
        return this->rows;
    }

    /// Brief: Returns the current number of columns.
    /// @return Number of columns.
    size_t get_cols(void) const {
        return this->cols;
    }

    /// Brief: Accesses a matrix element by row and column.
    /// @param row Zero-based row index.
    /// @param col Zero-based column index.
    /// @return Mutable reference to the requested element.
    float &at(size_t row, size_t col) {
        ASSERT_VAR(row < this->rows && col < this->cols,
            "Matrix::at(): Index (" << row << ", " << col
            << ") out of bounds for matrix ("
            << this->rows << "x" << this->cols << ").");

        size_t nCols = this->cols;
        return this->data[(row * nCols) + col];
    }

    /// Brief: Accesses an element of a const matrix.
    /// @param row Zero-based row index.
    /// @param col Zero-based column index.
    /// @return Const reference to the requested element.
    const float &at(size_t row, size_t col) const {
        ASSERT_VAR(row < this->rows && col < this->cols,
            "Matrix::at() const: Index (" << row << ", " << col
            << ") out of bounds for matrix ("
            << this->rows << "x" << this->cols << ").");

        size_t nCols = this->cols;
        return this->data[(row * nCols) + col];
    }

    /// Brief: Resizes the matrix and adjusts its internal storage.
    /// @param rows New number of rows.
    /// @param cols New number of columns.
    /// @return None.
    void resize(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols);
        return;
    }

    /// Brief: Fills every matrix element with the specified value.
    /// @param v Value used to fill the matrix.
    /// @return None.
    void fill(float v) {
        for(auto &i : this->data) {
            i = v;
        }
        return;
    }

    /// Brief: Initialises matrix elements using a uniform random distribution.
    /// @param min Lower bound of the distribution.
    /// @param max Upper bound of the distribution.
    /// @return None.
    void uniform_initialisation(float min, float max) {
        ASSERT_VAR(min <= max,
            "Matrix::uniform_initialisation(): Invalid range ["
            << min << ", " << max << "].");

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        for(auto &i : this->data) {
            i = dist(gen);
        }
        return;
    }

    /// Brief: Extracts a single column as a new matrix.
    /// @param col Zero-based column index.
    /// @return Matrix containing the selected column.
    Matrix get_col(size_t col) const {
        ASSERT_VAR(col < this->cols,
            "Matrix::get_col(): Column " << col
            << " out of bounds for " << this->cols << " columns.");

        Matrix colM(this->rows, 1);
        for(size_t i = 0; i < this->rows; i++) {
            colM.at(i, 0) = this->at(i, col);
        }
        return colM;
    }

    /// Brief: Extracts a single row as a new matrix.
    /// @param row Zero-based row index.
    /// @return Matrix containing the selected row.
    Matrix get_row(size_t row) const {
        ASSERT_VAR(row < this->rows,
            "Matrix::get_row(): Row " << row
            << " out of bounds for " << this->rows << " rows.");

        Matrix rowM(1, this->cols);
        for(size_t i = 0; i < this->cols; i++) {
            rowM.at(0, i) = this->at(row, i);
        }
        return rowM;
    }

    /// Brief: Appends another matrix vertically.
    /// @param m Matrix to append beneath this matrix.
    /// @return New matrix containing both matrices stacked vertically.
    Matrix append_rows(const Matrix &m) const {
        ASSERT_VAR(this->cols == m.cols,
            "Matrix::append_rows(): Column mismatch ("
            << this->cols << " vs " << m.cols << ").");

        Matrix result = *this;

        result.rows += m.rows;
        result.data.reserve(this->data.size() + m.data.size());
        result.data.insert(result.data.end(), m.data.begin(), m.data.end());
        return result;
    }

    /// Brief: Appends another matrix horizontally.
    /// @param m Matrix to append to the right.
    /// @return New matrix containing both matrices side by side.
    Matrix append_cols(const Matrix &m) const {
        ASSERT_VAR(this->rows == m.rows,
            "Matrix::append_cols(): Row mismatch ("
            << this->rows << " vs " << m.rows << ").");

        Matrix result(this->rows, this->cols + m.cols);

        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols + m.cols; c++) {
                if(c >= this->cols) {
                    result.at(r, c) = m.at(r, c - this->cols);
                } else {
                    result.at(r, c) = this->at(r, c);
                }
            }
        }

        return result;
    }

    /// Brief: Adds two matrices element-wise.
    /// @param rhs Matrix to add to this matrix.
    /// @return New matrix containing the element-wise sum.
    Matrix operator+(const Matrix &rhs) const {
        ASSERT_VAR(this->rows == rhs.rows && this->cols == rhs.cols,
            "Matrix::operator+(): Dimension mismatch ("
            << this->rows << "x" << this->cols
            << " vs " << rhs.rows << "x" << rhs.cols << ").");

        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] + rhs.data[i];
        }
        return result;
    }

    /// Brief: Subtracts two matrices element-wise.
    /// @param rhs Matrix to subtract from this matrix.
    /// @return New matrix containing the element-wise difference.
    Matrix operator-(const Matrix &rhs) const {
        ASSERT_VAR(this->rows == rhs.rows && this->cols == rhs.cols,
            "Matrix::operator-(): Dimension mismatch ("
            << this->rows << "x" << this->cols
            << " vs " << rhs.rows << "x" << rhs.cols << ").");

        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] - rhs.data[i];
        }
        return result;
    }

    /// Brief: Performs standard matrix multiplication.
    /// @param rhs Right-hand matrix operand.
    /// @return New matrix containing the matrix product.
    Matrix operator*(const Matrix &rhs) const {
        ASSERT_VAR(this->cols == rhs.rows,
            "Matrix::operator*(): Incompatible dimensions ("
            << this->rows << "x" << this->cols
            << " * " << rhs.rows << "x" << rhs.cols << ").");

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

    /// Brief: Multiplies two matrices element-wise using the Hadamard product.
    /// @param rhs Matrix to multiply element-wise.
    /// @return New matrix containing the element-wise product.
    Matrix hadamard(const Matrix &rhs) const {
        ASSERT_VAR(this->rows == rhs.rows && this->cols == rhs.cols,
            "Matrix::hadamard(): Dimension mismatch ("
            << this->rows << "x" << this->cols
            << " vs " << rhs.rows << "x" << rhs.cols << ").");

        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] * rhs.data[i];
        }
        return result;
    }

    /// Brief: Multiplies every element by a scalar.
    /// @param scalar Scalar multiplier.
    /// @return New matrix containing the scaled elements.
    Matrix scalar_multiply(float scalar) const {
        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }

    /// Brief: Returns the transpose of the matrix.
    /// @return New matrix with rows and columns swapped.
    Matrix transpose(void) const {
        Matrix result(this->cols, this->rows);

        for(size_t r = 0; r < this->rows; r++) {
            for(size_t c = 0; c < this->cols; c++) {
                result.at(c, r) = this->at(r, c);
            }
        }

        return result;
    }

    /// Brief: Applies an activation function element-wise to the matrix.
    /// @param act Pointer to a function accepting and returning a float.
    /// @return New matrix containing the activated values.
    Matrix activate(float (*act)(float)) const {
        ASSERT_VAR(act != nullptr,
            "Matrix::activate(): Activation function pointer is null.");

        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows * this->cols; i++) {
            result.data[i] = act(this->data[i]);
        }

        return result;
    }
};

#endif

