#include <iostream>
#include <random>
#include <vector>

#include "common.hpp"

struct Model {
    int vocab_size;
    std::mt19937 gen;
    matrix_type W;
    matrix_type dW;
    matrix_type xenc;
    matrix_type counts;
    matrix_type logits;
    matrix_type counts_sum;
    matrix_type counts_sum_inv;
    matrix_type probs;

    Model(size_t vocab_size, std::mt19937 gen) : vocab_size(vocab_size), gen(gen) {
        W = matrix_randn(vocab_size, vocab_size, gen);
    }

    matrix_type forward(const std::vector<int> &xs) {
        xenc = matrix_one_hot(xs, vocab_size);
        logits = matmul(xenc, W); // predict log-counts
        counts = exp(logits);     // Equivalent to the N matrix in bigram_model.cpp
        counts_sum = matrix_sum_rows(counts);
        counts_sum_inv = pow(counts_sum, -1.0f);
        probs = matmul_eltwise_broadcast(counts, counts_sum_inv);
        return probs;
    }

    void backward(matrix_type &dlogprobs) {
        matrix_type dprobs = matmul_eltwise(pow(probs, -1.0f), dlogprobs);
        matrix_type dcounts_sum_inv = matrix_sum_rows(matmul_eltwise(counts, dprobs));
        matrix_type dcounts0 = matmul_eltwise_broadcast(dprobs, counts_sum_inv);
        matrix_type dcounts_sum = pow(counts_sum, -2);
        for (size_t i = 0; i < dcounts_sum.size(); i++) {
            dcounts_sum[i][0] = -1.0 * dcounts_sum[i][0] * dcounts_sum_inv[i][0];
        }

        matrix_type dcounts_sum_bcast = matmul_eltwise_broadcast(matrix_ones_like(counts), dcounts_sum);
        matrix_type dcounts1 = matadd(dcounts0, dcounts_sum_bcast);

        matrix_type dlogits = matmul_eltwise(counts, dcounts1);
        matrix_type xenc_t = transpose(xenc);
        dW = matmul(xenc_t, dlogits);
    }
};

struct Loss {
    std::tuple<matrix_type, float> forward(matrix_type &probs, std::vector<int> &ys) {
        const size_t num = ys.size();
        matrix_type logprobs = log(probs);

        // In pytorch: -logprobs[torch.arange(num), ys].mean()
        float loss = 0;
        for (size_t i = 0; i < num; i++) {
            loss += -logprobs[i][ys[i]];
        }
        loss /= num;

        return {logprobs, loss};
    }

    matrix_type backward(matrix_type &logprobs, std::vector<int> &ys) {
        const size_t num = ys.size();
        matrix_type dlogprobs = matrix_zeros_like(logprobs);
        for (size_t i = 0; i < num; i++) {
            dlogprobs[i][ys[i]] = -1.0 / num;
        }
        return dlogprobs;
    }
};

int main(void) {
    std::string filename = "names.txt";

    auto [words, stoi, itos, chars] = read_file(filename);

    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(INT_MAX);

    auto vocab_size = itos.size();

    std::cout << "vocab_size=" << vocab_size << std::endl;
    std::cout << "words=" << words.size() << std::endl;

    std::vector<int> xs;
    std::vector<int> ys;

    for (const auto &w : words) {
        std::vector<char> chs;
        chs.push_back('.');
        for (const auto &ch : w) {
            chs.push_back(ch);
        }
        chs.push_back('.');

        for (size_t i = 0; i < chs.size() - 1; i++) {
            const auto ch1 = chs[i];
            const auto ch2 = chs[i + 1];
            const auto ix1 = stoi[ch1];
            const auto ix2 = stoi[ch2];
            xs.push_back(ix1);
            ys.push_back(ix2);
        }
    }

    std::cout << "xs.size()=" << xs.size() << ", ys.size()=" << ys.size() << std::endl;

    Model model(vocab_size, g);
    Loss loss_module;

    float lr = 10.0;
    size_t num_steps = 1000;

    for (size_t k = 0; k < num_steps; k++) {
        matrix_type probs = model.forward(xs);
        auto [logprobs, loss] = loss_module.forward(probs, ys);

        if (k % 10 == 0 || k == num_steps - 1) {
            std::cout << "step: " << k << "/" << num_steps - 1 << ": lr=" << lr << ", loss=" << loss << std::endl;
        }

        matrix_type dlogprobs = loss_module.backward(logprobs, ys);
        model.backward(dlogprobs);

        // Weight update
        for (size_t r = 0; r < vocab_size; r++) {
            for (size_t c = 0; c < vocab_size; c++) {
                model.W[r][c] += -lr * model.dW[r][c];
            }
        }
    }

    std::cout << std::endl << "Eval:" << std::endl;

    for (size_t i = 0; i < 5; i++) {
        std::vector<char> out;
        int ix = 0;

        while (true) {
            matrix_type probs = model.forward({ix});
            // ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            std::discrete_distribution<> dist(probs.front().begin(), probs.front().end());
            ix = dist(g);
            std::cout << itos[ix];
            out.push_back(itos[ix]);
            if (ix == 0) {
                std::cout << std::endl;
                break;
            }
        }
    }
    return 0;
}