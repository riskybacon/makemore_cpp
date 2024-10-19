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

matrix_type matrix_empty(size_t rows, size_t cols) { return matrix_type(rows, vector_type(cols)); }

matrix_type matrix_empty_like(const matrix_type &mat) {
    return matrix_type(mat.size(), vector_type(mat.front().size()));
}

matrix_type matrix_zeros(size_t rows, size_t cols) {
    matrix_type matrix(rows, vector_type(cols, 0));
    return matrix;
}

matrix_type matrix_zeros_like(const matrix_type &mat) { return matrix_zeros(mat.size(), mat.front().size()); }

matrix_type matrix_ones(size_t rows, size_t cols) {
    matrix_type matrix(rows, vector_type(cols, 1));
    return matrix;
}

matrix_type matrix_ones_like(const matrix_type &mat) { return matrix_ones(mat.size(), mat.front().size()); }

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

// Function to fill a 2D vector with random numbers from a normal distribution
matrix_type matrix_randn(size_t rows, size_t cols, std::mt19937 &gen, float mean = 0.0, float variance = 1.0) {
    // Create a normal distribution with the given mean and variance (standard deviation = sqrt(variance))
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

matrix_type matrix_one_hot(const std::vector<int> &xs, size_t num_classes) {
    auto mat = matrix_zeros(xs.size(), num_classes);

    for (size_t i = 0; i < xs.size(); i++) {
        int idx = xs[i];
        mat[i][idx] = 1.0;
    }

    return mat;
}

matrix_type matrix_one_hot(const std::initializer_list<int> &&xs, size_t num_classes) {
    auto mat = matrix_zeros(xs.size(), num_classes);

    size_t i = 0;
    for (const auto &idx : xs) {
        mat[i][idx] = 1.0f;
    }

    return mat;
}

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

matrix_type matmul_eltwise_broadcast(const matrix_type &lhs, const vector_type &rhs) {
    const size_t lhs_rows = lhs.size();
    const size_t lhs_cols = lhs[0].size();
    const size_t rhs_rows = rhs.size();

    if (lhs_rows != rhs_rows) {
        throw std::invalid_argument("Matrix dimensions do not match row dimenstions for broadcast");
    }

    auto out = matrix_empty_like(lhs);

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < lhs_cols; ++j) {
            out[i][j] = lhs[i][j] * rhs[i];
        }
    }

    return out;
}

matrix_type matadd_eltwise(const matrix_type &lhs, const matrix_type &rhs) {
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

vector_type matrix_sum_rows(const matrix_type &mat) {
    const size_t rows = mat.size();
    const size_t cols = mat[0].size();

    vector_type row_sums(rows);

    for (size_t i = 0; i < rows; ++i) {
        row_sums[i] = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            row_sums[i] += mat[i][j];
        }
    }

    return row_sums;
}

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

vector_type pow(const vector_type &lhs, float y) {
    vector_type result(lhs.size());
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = std::pow(lhs[i], y);
    }

    return result;
}

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
