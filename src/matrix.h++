#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>


/**
 * NOTE: This is a CPU-based library
 * Its intended for use with small perceptrons
 * std::thread could be used for larger matricies
 */


namespace matrix {
    template <typename T>
    class Matrix {
        private:
        size_t rows, cols;

        //For transpose and mmul (operations can't be done in place)
        //This avoids constant reallocation
        std::vector<T> data;
        public:

        /**
         * @brief Construct an empty matrix.
         */
        Matrix(void) {
            this->rows = 0;
            this->cols = 0;
        }

        /**
         * @brief Construct a matrix with the specified dimensions.
         *
         * @param rows Number of rows.
         * @param cols Number of columns.
         */
        Matrix(size_t rows, size_t cols) {
            this->rows = rows;
            this->cols = cols;
            data.resize(rows * cols);
        }

        /**
         * @brief Construct a matrix by copying values from a vector.
         *
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param data Vector containing the matrix values.
         */
        Matrix(size_t rows, size_t cols, std::vector<T> &data) {
            this->rows = rows;
            this->cols = cols;
            this->data = data;
        }

       /**
        * @brief Replace the matrix values with those in a vector.
        *
        * @param vec Vector to copy from.
        *
        * @return void
        */
        void fill_vector(std::vector<T> &vec) {
            this->data = vec;
            return;
        } 


       /**
        * @brief Get a constant reference to the matrix's values.
        *
        * @return const std::vector<T>& Constant reference to the internal values.
        */
        const std::vector<T> &get_vector(void) {
            return this->data;;
        } 


        /**
         * @brief Access a matrix element.
         * 
         * @param row Zero-based row index.
         * @param col Zero-based column index.
         * 
         * @return T& Reference to the element.
         */
        inline T& at(size_t row, size_t col) {
            return this->data[row * this->cols + col];
        }

        /**
         * @brief Access a matrix element without allowing modification.
         *
         * @param row Zero-based row index.
         * @param col Zero-based column index.
         * @return const T& Constant reference to the element.
         */
        inline const T& at(size_t row, size_t col) const {
            return this->data[row * this->cols + col];
        }

        /**
         * @brief Resize the matrix to the specified dimensions.
         * 
         * @param rows New number of rows.
         * @param cols New number of columns.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &redimension(size_t rows, size_t cols) {

            this->rows = rows;
            this->cols = cols;
            this->data.resize(rows * cols);
            return *this;
        }        


        /**
         * @brief Initialize the matrix using the Xavier uniform distribution.
         *
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &xavier_uniform() {

            constexpr float numerator = 6.0f;

            const float limit = std::sqrt(
                numerator / static_cast<float>(this->rows + this->cols)
            );

            static thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(-limit, limit);

            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = dist(rng);
            }

            return *this;
        }


        /**
         * @brief Fill every element of the matrix with a value.
         * 
         * @param value Value to assign to each element.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &fill(T value) {

            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = value;
            }

            return *this;
        }   


        /**
         * @brief Multiply this matrix by another matrix and return the result.
         * 
         * @param rhs Right-hand matrix; its row count must match this matrix's column count.
         * 
         * @return Matrix<T> Product matrix.
         */
        Matrix<T> operator*(Matrix<T> &rhs) {
            Matrix<T> result(this->rows, rhs.cols);

            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < rhs.cols; j++) {
                    result.at(i, j) = T{};
                    for(size_t k = 0; k < this->cols; k++) {
                        result.at(i, j) += this->at(i, k) * rhs.at(k, j);
                    }
                }
            }

            return result;
        }

        /**
         * @brief Add another matrix element-wise and return the result.
         *
         * @param rhs Right-hand matrix with the same dimensions as this matrix.
         *
         * @return Matrix<T> Sum matrix.
         */
        Matrix<T> operator+(Matrix<T> &rhs) {
            Matrix<T> result(this->rows, this->cols);
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                result.data[i] = this->data[i] + rhs.data[i];
            }

            return result;
        }

        /**
         * @brief Append a column vector to this matrix.
         *
         * @param column Column vector to append; it must have the same number of rows.
         *
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &append_column(Matrix<T> &column) {

            std::vector<T> appended;
            appended.reserve(this->rows * (this->cols + 1));
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    appended.push_back(this->at(i, j));
                }
                appended.push_back(column.at(i, 0));
            }

            this->data = std::move(appended);
            this->cols++;
            return *this;
        }


        


        /**
         * @brief Multiply two matrices into this matrix, optionally transposing either input.
         * 
         * @param x Left-hand input matrix.
         * @param w Right-hand input matrix.
         * @param transposeX Whether to transpose the left-hand input.
         * @param transposeW Whether to transpose the right-hand input.
         * 
         * @return Matrix<T>& Reference to this matrix containing the product.
         */
        Matrix<T> &multiply(Matrix<T> &x, Matrix<T> &w, bool transposeX, bool transposeW) {
            //Multiply

            size_t xr = transposeX ? x.cols : x.rows;
            size_t xc = transposeX ? x.rows : x.cols;

            size_t wr = transposeW ? w.cols : w.rows;
            size_t wc = transposeW ? w.rows : w.cols;

            for(size_t i = 0; i < xr; i++) {
                for(size_t j = 0; j < wc; j++) {

                    this->at(i, j) = 0;
                    for(size_t k = 0; k < xc; k++) {

                        T xVal = transposeX ? x.at(k, i) : x.at(i, k);
                        T wVal = transposeW ? w.at(j, k) : w.at(k, j);

                        this->at(i, j) += xVal * wVal;
                    }
                }
            }
            return *this;
        }


        /**
         * @brief Add another matrix element-wise to this matrix.
         * 
         * @param rhs Matrix to add; it must have the same number of elements.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &add(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] += rhs.data[i];
            }
            return *this;
        }

        /**
         * @brief Subtract another matrix element-wise from this matrix.
         * 
         * @param rhs Matrix to subtract; it must have the same number of elements.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &subtract(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] -= rhs.data[i];
            }
            return *this;
        }

        /**
         * @brief Multiply this matrix element-wise by another matrix.
         * 
         * @param rhs Matrix to multiply element-wise; it must have the same number of elements.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &hadamard(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] *= rhs.data[i];
            }
            return *this;
        }


        /**
         * @brief Get the number of rows in the matrix.
         * 
         * @return size_t Number of rows.
         */
        size_t get_rows(void) {
            return this->rows;
        }


        /**
         * @brief Get the number of columns in the matrix.
         * 
         * @return size_t Number of columns.
         */
        size_t get_cols(void) {
            return this->cols;
        }

        /**
         * @brief Multiply every element of the matrix by a scalar.
         * 
         * @param scalar Scalar to multiply by.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &scalar_multiply(T scalar) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] *= scalar;
            }
            return *this;
        }


        /**
         * @brief Add the first row of another matrix to each row of this matrix.
         * 
         * @param matrix Matrix whose first row is added to each row of this matrix.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &add_row(Matrix<T> &matrix) {
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    this->at(i, j) += matrix.at(0, j);
                }
            }
            return *this;
        }


        /**
         * @brief Append the matrix dimensions and values to a file.
         * 
         * @param fileName Path of the file to append to.
         * 
         * @return bool True if writing to the file succeeds; otherwise, false.
         */
        bool append_to_file(std::string fileName) {
            std::ofstream file(fileName, std::ios::app);
            if(!file) {
                return false;
            }
            file << this->rows << "\n" << this->cols << "\n";
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    file << this->at(i, j) << " ";
                }
            }
            file << "\n";
            return file.good(); //Make sure file is ok
        }


        /**
         * @brief Read matrix dimensions and values from a file stream.
         * 
         * @param file Input stream containing rows, columns, and matrix values.
         * 
         * @return bool True if the stream is valid at entry; otherwise, false.
         */
        bool read_float_file_fstream(std::ifstream &file) {
            if(!file) {
                return false;
            }
            

            //Get dimensions
            std::string line;
            std::getline(file, line);
            size_t rows = std::stoull(line);
            std::getline(file, line);
            size_t cols = std::stoull(line);
            this->rows = rows;
            this->cols = cols;
            this->data.resize(rows * cols);

            //Get data
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    std::getline(file, line, ' ');
                    this->data[i * cols + j] = std::stof(line);
                }
            }
            //Skip \n
            std::getline(file, line);

            return true;
        }


        /**
         * @brief Add each row of another matrix to the first row of this matrix.
         * 
         * @param matrix Matrix whose rows are accumulated.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &sum_rows(Matrix<T> &matrix) {
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    this->at(0, j) += matrix.at(i, j);
                }
            }
            return *this;
        }
        

        /**
         * @brief Apply the ReLU activation element-wise.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_relu(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = (this->data[i] > 0) * this->data[i];
            }
            return *this;
        }
        /**
         * @brief Replace each element with its ReLU derivative.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_derivative_relu(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = (this->data[i] > 0);
            }
            return *this;
        }

        /**
         * @brief Apply the sigmoid activation element-wise.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_sigmoid(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = 1/(1 + exp(-1 * this->data[i]));
            }
            return *this;
        }

        /**
         * @brief Replace each element with its sigmoid derivative.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_derivative_sigmoid(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {

                this->data[i] = this->data[i] * (1 - this->data[i]);
            }
            return *this;
        }
        /**
         * @brief Copy a column from this matrix into a column vector.
         *
         * @param col Zero-based index of the column to retrieve.
         *
         * @return Matrix<T> Column vector with the same number of rows.
         */
        Matrix<T> get_column(size_t col) const {


            Matrix<T> result(this->rows, 1);

            for(size_t i = 0; i < this->rows; i++) {
                result.at(i, 0) = this->at(i, col);
            }

            return result;
        }

        /**
         * @brief Apply the hyperbolic tangent activation element-wise.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_tanh(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = tanh(this->data[i]);
            }
            return *this;
        }

        /**
         * @brief Replace each element with its hyperbolic tangent derivative.
         * 
         * @return Matrix<T>& Reference to this matrix.
         */
        Matrix<T> &activate_derivative_tanh(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {

                this->data[i] = 1 - (this->data[i] * this->data[i]);
            }
            return *this;
        }
    };
}



#endif
