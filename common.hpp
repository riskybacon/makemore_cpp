#pragma once
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using matrix_type = std::vector<std::vector<float>>;
using vector_type = std::vector<float>;

/**
 * @brief Reads the content of a file and returns various data structures based on its contents.
 *
 * This function reads the specified file and processes its contents to generate four outputs:
 * 1. A vector of strings, where each string represents a line/word in the file.
 * 2. An unordered map (char to int) representing the mapping from characters to indices.
 * 3. An unordered map (int to char) representing the mapping from indices to characters.
 * 4. A vector of characters, representing unique characters in the file. Each element's index
 *    corresponds to the indices in the two unordered_maps
 *
 * @param filename The name of the file to read.
 *
 * @return A tuple containing:
 *         - words
 *         - map of characters to indices
 *         - map of indices to characters
 *         - A vector of characters
 */
std::tuple<std::vector<std::string>, std::unordered_map<char, int>, std::unordered_map<int, char>, std::vector<char>>
read_file(const std::string &filename) {
    std::string word;
    std::ifstream file(filename);
    std::vector<std::string> words;
    std::unordered_set<char> chars_set;

    while (!file.eof()) {
        file >> word;
        words.push_back(word);
        for (const auto &ch : word) {
            chars_set.insert(ch);
        }
    }

    std::vector<char> chars = std::vector<char>(begin(chars_set), end(chars_set));
    std::sort(begin(chars), end(chars));
    chars.insert(begin(chars), '.');

    // build the vocabulary of characters and mappings to/from integers
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;

    for (int i = 0; i < chars.size(); i++) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }

    return {words, stoi, itos, chars};
}

/**
 * @brief Creates an empty matrix of specified dimensions.
 *
 * This function generates a matrix (2D vector) of the specified size,
 * with all elements uninitialized.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A matrix (vector of vectors of floats) with dimensions (rows x cols),
 *         where all elements are not initialized.
 */
matrix_type matrix_empty(size_t rows, size_t cols) { return matrix_type(rows, vector_type(cols)); }

/**
 * @brief Creates an empty matrix with the same dimensions as the input matrix.
 *
 * This function generates a new matrix (2D vector) that has the same number
 * of rows and columns as the input matrix `mat`, with all elements unintialized.
 *
 * @param mat The input matrix whose dimensions will be used to create the new matrix.
 *
 * @return A matrix (vector of vectors of floats) with the same dimensions as `mat`,
 *         where all elements are not initialized.
 */
matrix_type matrix_empty_like(const matrix_type &mat) {
    return matrix_type(mat.size(), vector_type(mat.front().size()));
}

/**
 * @brief Creates a matrix of zeros of specified dimensions.
 *
 * This function generates a matrix (2D vector) of the specified size,
 * with all elements initialized to zero.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A matrix (vector of vectors of floats) with dimensions (rows x cols),
 *         where all elements are initialized to 0.0f.
 */
matrix_type matrix_zeros(size_t rows, size_t cols) {
    matrix_type matrix(rows, vector_type(cols, 0));
    return matrix;
}

/**
 * @brief Creates a matrix of zeros with the same dimensions as the input matrix.
 *
 * This function generates a new matrix (2D vector) that has the same number
 * of rows and columns as the input matrix `mat`, with all elements intialized to zero.
 *
 * @param mat The input matrix whose dimensions will be used to create the new matrix.
 *
 * @return A matrix (vector of vectors of floats) with the same dimensions as `mat`,
 *         where all elements are initialized to zero.
 */
matrix_type matrix_zeros_like(const matrix_type &mat) { return matrix_zeros(mat.size(), mat.front().size()); }

/**
 * @brief Creates a matrix of ones of specified dimensions.
 *
 * This function generates a matrix (2D vector) of the specified size,
 * with all elements initialized to one.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A matrix (vector of vectors of floats) with dimensions (rows x cols),
 *         where all elements are initialized to one.
 */
matrix_type matrix_ones(size_t rows, size_t cols) {
    matrix_type matrix(rows, vector_type(cols, 1));
    return matrix;
}

/**
 * @brief Creates a matrix of ones with the same dimensions as the input matrix.
 *
 * This function generates a new matrix (2D vector) that has the same number
 * of rows and columns as the input matrix `mat`, with all elements intialized to one.
 *
 * @param mat The input matrix whose dimensions will be used to create the new matrix.
 *
 * @return A matrix (vector of vectors of floats) with the same dimensions as `mat`,
 *         where all elements are initialized to one.
 */
matrix_type matrix_ones_like(const matrix_type &mat) { return matrix_ones(mat.size(), mat.front().size()); }

/**
 * @brief Computes the transpose of a matrix.
 *
 * This function takes an input matrix and returns its transpose.
 * The transpose of a matrix is formed by swapping its rows and columns.
 * For a matrix `mat` with dimensions (m x n), the resulting matrix will have
 * dimensions (n x m), where each element at position (i, j) in the original
 * matrix is placed at position (j, i) in the transposed matrix.
 *
 * @param mat The input matrix (vector of vectors of floats) to be transposed.
 *
 * @return A matrix (vector of vectors of floats) representing the transpose of `mat`.
 *         The resulting matrix will have dimensions (n x m), where `m` and `n` are the
 *         number of rows and columns in the input matrix, respectively.
 */
matrix_type transpose(const matrix_type &mat) {
    const size_t in_rows = mat.size();
    const size_t in_cols = mat.front().size();
    const size_t out_rows = in_cols;
    const size_t out_cols = in_rows;
    auto out = matrix_empty(out_rows, out_cols);

    for (size_t i = 0; i < out_rows; i++) {
        for (size_t j = 0; j < out_cols; j++) {
            out[i][j] = mat[j][i];
        }
    }

    return out;
}

/**
 * @brief Creates a matrix with elements drawn from a normal distribution.
 *
 * This function generates a matrix of the specified dimensions (`rows` x `cols`),
 * where each element is sampled from a normal (Gaussian) distribution with the given
 * mean and variance. The distribution is generated using the provided random number
 * generator `gen`.
 *
 * @param rows The number of rows in the resulting matrix.
 * @param cols The number of columns in the resulting matrix.
 * @param gen  A Mersenne Twister random number generator used for drawing random samples.
 * @param mean The mean of the normal distribution. Default value is 0.0.
 * @param variance The variance of the normal distribution. Default value is 1.0.
 *
 * @return A matrix (vector of vectors of floats) with dimensions (`rows` x `cols`),
 *         where each element is drawn from N(mean, variance).
 */
matrix_type matrix_randn(size_t rows, size_t cols, std::mt19937 &gen, float mean = 0.0, float variance = 1.0) {
    std::normal_distribution<float> dist(mean, std::sqrt(variance));
    auto matrix = matrix_empty(rows, cols);

    // Fill the 2D vector with random numbers
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

/**
 * @brief Converts a vector of indices into a one-hot encoded matrix.
 *
 * This function takes a vector of index values and returns a one-hot encoded matrix with
 * dimensions (`xs.size()` x `num_classes`). Each row in the matrix corresponds to an element
 * in the input vector `xs`, with a 1 placed at the index specified by the value in `xs`,
 * and 0s elsewhere.
 *
 * For example, given an input vector `xs = {1, 0, 2}` and `num_classes = 3`, the resulting matrix
 * will be:
 *
 * \code
 * 0 1 0
 * 1 0 0
 * 0 0 1
 * \endcode
 *
 * @param xs A vector of integers representing the indices to be one-hot encoded.
 *           Each element in `xs` must be in the range [0, `num_classes` - 1].
 * @param num_classes The number of classes (columns) in the resulting one-hot matrix.
 *
 * @return A matrix (vector of vectors of floats) with dimensions (`xs.size()` x `num_classes`),
 *         where each row contains a one-hot encoding of the corresponding element in `xs`.
 *
 * @throws std::out_of_range if any element in `xs` is greater than or equal to `num_classes`.
 */
matrix_type matrix_one_hot(const std::vector<int> &xs, size_t num_classes) {
    auto mat = matrix_zeros(xs.size(), num_classes);

    for (size_t i = 0; i < xs.size(); i++) {
        int idx = xs[i];
        mat[i].at(idx) = 1.0;
    }

    return mat;
}

/**
 * @brief Multiplies two matrices and returns the resulting matrix.
 *
 * This function performs matrix multiplication on two input matrices `lhs` and `rhs`.
 * The number of columns in the left-hand side matrix (`lhs`) must match the number of
 * rows in the right-hand side matrix (`rhs`) for the multiplication to be valid.
 *
 * The resulting matrix will have dimensions equal to the number of rows in `lhs`
 * and the number of columns in `rhs`.
 *
 * @param lhs The left-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (m x n), where `m` is the number of rows and `n` is the number of columns.
 * @param rhs The right-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (n x p), where `n` is the number of rows (matching `lhs` columns),
 *            and `p` is the number of columns.
 *
 * @return A matrix (vector of vectors of floats) that represents the product of `lhs` and `rhs`.
 *         The result will have dimensions (m x p), where `m` is the number of rows in `lhs`
 *         and `p` is the number of columns in `rhs`.
 *
 * @throws std::invalid_argument if the number of columns in `lhs` does not match the number of rows in `rhs`.
 */
matrix_type matmul(const matrix_type &lhs, const matrix_type &rhs) {
    const size_t lhs_rows = lhs.size();
    const size_t lhs_cols = lhs[0].size();
    const size_t rhs_rows = rhs.size();
    const size_t rhs_cols = rhs[0].size();

    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    auto result = matrix_zeros(lhs_rows, rhs_cols);

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < rhs_cols; ++j) {
            for (size_t k = 0; k < lhs_cols; ++k) {
                result[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }

    return result;
}

/**
 * @brief Multiplies two matrices element-by-element.
 *
 * Performs an elementwise matrix multiplication, also known as the Hadamard
 * product. The `lhs` and `rhs` matrices must have the same dimensions, and
 * are multiplied together by multiplying their corresponding elements.
 * Unlike traditional matrix multiplication, which involves dot products of
 * rows and columns, the Hadamard product is simpler and works on an
 * element-by-element basis.
 *
 * The resulting matrix will have dimensions equal to the number of rows in `lhs`
 * and the number of columns in `rhs`.
 *
 * @param lhs The left-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (m x n), where `m` is the number of rows
 *            and `n` is the number of columns.
 * @param rhs The right-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (m x n)
 *
 * @return A matrix (vector of vectors of floats) that represents the
 *         element-wise product of `lhs` and `rhs`.
 *         The result will have dimensions (m x n), where `m` is the number of
 *         rows in `lhs` and `rhs` and `n` is the number of columns in `lhs`
 *         and `rhs`
 *
 * @throws std::invalid_argument if the number of columns in `lhs` does not match the number of rows in `rhs`.
 */
matrix_type matmul_eltwise(const matrix_type &lhs, const matrix_type &rhs) {
    const size_t lhs_rows = lhs.size();
    const size_t lhs_cols = lhs[0].size();
    const size_t rhs_rows = rhs.size();
    const size_t rhs_cols = rhs[0].size();

    if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
        throw std::invalid_argument("Matrix dimensions do not match row dimensions for eltwise multiplication");
    }

    auto out = matrix_empty_like(lhs);

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < lhs_cols; ++j) {
            out[i][j] = lhs[i][j] * rhs[i][j];
        }
    }

    return out;
}

/**
 * @brief Multiplies a matrix and a vector element-by-element.
 *
 * This is an element-wise matrix multiplication, with the `rhs` matrix constructed
 * by treating it like a column matrix that is repeated for the same number of columns
 * that are in the `lhs` matrix.
 *
 * The resulting matrix will have the same dimensions as `lhs`.
 *
 * @param lhs The left-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (m x n), where `m` is the number of rows
 *            and `n` is the number of columns.
 * @param rhs The right-hand side vector (vector of floats).
 *            Must have dimensions (m x 1)
 *
 * @return A matrix (vector of vectors of floats) that represents the
 *         element-wise product of `lhs` and `rhs`.
 *         The result will have dimensions (m x n), where `m` is the number of
 *         rows in `lhs` and `rhs` and `n` is the number of columns in `lhs`
 *         and `rhs`
 *
 * @throws std::invalid_argument if the number of rows in `lhs` does not match the number of rows in `rhs`.
 */
matrix_type matmul_eltwise_broadcast(const matrix_type &lhs, const matrix_type &rhs) {
    const size_t lhs_rows = lhs.size();
    const size_t lhs_cols = lhs[0].size();
    const size_t rhs_rows = rhs.size();
    const size_t rhs_cols = rhs[0].size();

    if (rhs_cols != 1) {
        throw std::invalid_argument("rhs_cols != 1");
    }

    if (lhs_rows != rhs_rows) {
        throw std::invalid_argument("Matrix dimensions do not match row dimenstions for broadcast");
    }

    auto out = matrix_empty_like(lhs);

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < lhs_cols; ++j) {
            out[i][j] = lhs[i][j] * rhs[i][0];
        }
    }

    return out;
}

/**
 * @brief Performs elementwise addition of two matrices.
 *
 * This function takes two matrices (`lhs` and `rhs`) of the same dimensions and
 * returns a new matrix where each element is the sum of the corresponding elements
 * from the input matrices.
 *
 * For two matrices `lhs` and `rhs` of dimensions (m x n), the resulting matrix
 * will also have dimensions (m x n), where each element is calculated as:
 *
 * \f[
 * C_{ij} = A_{ij} + B_{ij}
 * \f]
 *
 * @param lhs The left-hand side matrix (vector of vectors of floats).
 *            Must have the same dimensions as `rhs`.
 * @param rhs The right-hand side matrix (vector of vectors of floats).
 *            Must have the same dimensions as `lhs`.
 *
 * @return A matrix (vector of vectors of floats) that contains the elementwise sum
 *         of `lhs` and `rhs`, with the same dimensions as the input matrices.
 *
 * @throws std::invalid_argument if the dimensions of `lhs` and `rhs` do not match.
 */
matrix_type matadd(const matrix_type &lhs, const matrix_type &rhs) {
    const size_t lhs_rows = lhs.size();
    const size_t lhs_cols = lhs[0].size();
    const size_t rhs_rows = rhs.size();
    const size_t rhs_cols = rhs[0].size();

    if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
        throw std::invalid_argument("Matrix dimensions do not match row dimensions for eltwise addition");
    }

    auto out = matrix_empty_like(lhs);

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < lhs_cols; ++j) {
            out[i][j] = lhs[i][j] + rhs[i][j];
        }
    }

    return out;
}

/**
 * @brief Computes the sum of each row in a matrix and returns a column matrix.
 *
 * This function takes an input matrix and computes the sum of each row.
 * It returns a new \( N \times 1 \) matrix (column matrix) where \( N \) is the number
 * of rows in the input matrix. Each element in the output matrix represents
 * the sum of the corresponding row in the input matrix.
 *
 * For an input matrix `mat` of dimensions (m x n), the resulting column matrix
 * will have dimensions (m x 1), where each element is calculated as:
 *
 * \f[
 * C_i = \sum_{j=0}^{n-1} A_{ij}
 * \f]
 *
 * @param mat The input matrix (vector of vectors of floats) to be summed across rows.
 *
 * @return A column matrix of size \( m \), where each element is the sum of the
 *         corresponding row in the input matrix.
 */
matrix_type matrix_sum_rows(const matrix_type &mat) {
    const size_t rows = mat.size();
    const size_t cols = mat[0].size();

    matrix_type out = matrix_zeros(rows, 1);

    for (size_t i = 0; i < rows; ++i) {
        out[i][0] = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            out[i][0] += mat[i][j];
        }
    }

    return out;
}

/**
 * @brief Applies the exponential function to each element of a matrix.
 * 
 * This function takes an input matrix and returns a new matrix of the same dimensions, 
 * where each element is the result of applying the exponential function (e^x) to the 
 * corresponding element in the input matrix.
 * 
 * For an input matrix `lhs` of dimensions (m x n), the resulting matrix will also 
 * have dimensions (m x n), where each element is calculated as:
 * 
 * \f[
 * C_{ij} = e^{A_{ij}}
 * \f]
 * 
 * @param lhs The input matrix (vector of vectors of floats) to be exponentiated.
 * 
 * @return A matrix (vector of vectors of floats) of the same dimensions as `lhs`, 
 *         where each element is the exponential of the corresponding element in `lhs`.
 */
matrix_type exp(const matrix_type &lhs) {
    const size_t rows = lhs.size();
    const size_t cols = lhs[0].size();

    auto result = matrix_zeros_like(lhs);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = std::exp(lhs[i][j]);
        }
    }

    return result;
}

/**
 * @brief Raises each element of a matrix to the power of a given exponent.
 * 
 * This function takes an input matrix and returns a new matrix of the same dimensions, 
 * where each element is raised to the power of the provided exponent `y`. 
 * The operation performed is equivalent to applying the `std::pow()` function to each 
 * element in the input matrix.
 * 
 * For an input matrix `lhs` of dimensions (m x n) and an exponent `y`, the resulting 
 * matrix will also have dimensions (m x n), where each element is calculated as:
 * 
 * \f[
 * C_{ij} = A_{ij}^y
 * \f]
 * 
 * @param lhs The input matrix (vector of vectors of floats) whose elements will be exponentiated.
 * @param y The exponent to which each element in the matrix `lhs` will be raised.
 * 
 * @return A matrix (vector of vectors of floats) of the same dimensions as `lhs`, 
 *         where each element is the result of raising the corresponding element in `lhs` to the power `y`.
 */
matrix_type pow(const matrix_type &lhs, float y) {
    const size_t rows = lhs.size();
    const size_t cols = lhs[0].size();
    auto out = matrix_empty(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out[i][j] = std::pow(lhs[i][j], y);
        }
    }

    return out;
}

/**
 * @brief Applies the natural logarithm to each element of a matrix.
 * 
 * This function takes an input matrix and returns a new matrix of the same dimensions, 
 * where each element is transformed by the natural logarithm function (`std::log()`).
 * 
 * For an input matrix `lhs` of dimensions (m x n), the resulting matrix will also 
 * have dimensions (m x n), where each element is calculated as:
 * 
 * \f[
 * C_{ij} = \log(A_{ij})
 * \f]
 * 
 * @param lhs The input matrix (vector of vectors of floats) whose elements will have the 
 *            natural logarithm applied.
 * 
 * @return A matrix (vector of vectors of floats) of the same dimensions as `lhs`, 
 *         where each element is the natural logarithm of the corresponding element in `lhs`.
 * 
 * @note The function assumes that all elements in the input matrix are positive, as 
 *       the natural logarithm is undefined for non-positive values.
 */
matrix_type log(const matrix_type &lhs) {
    const size_t rows = lhs.size();
    const size_t cols = lhs[0].size();
    auto out = matrix_empty(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out[i][j] = std::log(lhs[i][j]);
        }
    }

    return out;
}
