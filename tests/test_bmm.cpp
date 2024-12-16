// File: tests/test_bmm.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "catch.hpp"
#include "bmm.h"
#include "bmm_blocking.h"

using namespace std;
using namespace std::chrono;

void generate_random_matrix(vector<double> &matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

double time_function(void (*func)(const double*, const double*, double*, int, int, int, int),
                     const double *a, const double *b, double *c,
                     int batch_dim, int a_rows, int b_cols, int a_cols) {
    auto start = high_resolution_clock::now();
    func(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_blocked_function(void (*func)(const double*, const double*, double*, int, int, int, int, int),
                             const double *a, const double *b, double *c,
                             int batch_dim, int a_rows, int b_cols, int a_cols, int block_size) {
    auto start = high_resolution_clock::now();
    func(a, b, c, batch_dim, a_rows, b_cols, a_cols, block_size);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    return elapsed.count();
}

void batch_matrix_multiply_wrapper(const double *a, const double *b, double *c,
                                   int batch_dim, int a_rows, int b_cols, int a_cols) {
    batch_matrix_multiply(a, b, c, batch_dim, a_rows, b_cols, a_cols);
}

void blocked_matrix_multiply_wrapper(const double *a, const double *b, double *c,
                                     int batch_dim, int a_rows, int b_cols, int a_cols, int block_size) {
    blocked_matrix_multiply(a, b, c, batch_dim, a_rows, b_cols, a_cols, block_size);
}

TEST_CASE("Batch Matrix Multiplication Timing", "[bmm]") {
    const int batch_dim = 10;
    const int a_rows = 512;
    const int b_cols = 512;
    const int a_cols = 512;

    vector<double> a(batch_dim * a_rows * a_cols);
    vector<double> b(batch_dim * a_cols * b_cols);
    vector<double> c(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    SECTION("Timing batch_matrix_multiply") {
        double time_taken = time_function(batch_matrix_multiply_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
        cout << "Time taken by batch_matrix_multiply: " << time_taken << " seconds" << endl;
    }

    SECTION("Timing blocked_matrix_multiply with different block sizes") {
        vector<int> block_sizes = {16, 32, 64, 128};
        for (int block_size : block_sizes) {
            fill(c.begin(), c.end(), 0.0); // Reset the result matrix
            double time_taken = time_blocked_function(blocked_matrix_multiply_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, block_size);
            cout << "Time taken by blocked_matrix_multiply with block size " << block_size << ": " << time_taken << " seconds" << endl;
        }
    }
}