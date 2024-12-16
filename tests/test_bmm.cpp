// File: tests/test_bmm.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "catch.hpp"
#include "bmm.h"
#include "bmm_blocking.h"
#include "bmm_omp.h"

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

void bmm_blocked_wrapper(const double *a, const double *b, double *c,
                                     int batch_dim, int a_rows, int b_cols, int a_cols, int block_size) {
    bmm_blocked(a, b, c, batch_dim, a_rows, b_cols, a_cols, block_size);
}

void bmm_omp_simple_wrapper(const double *a, const double *b, double *c,
                            int batch_dim, int a_rows, int b_cols, int a_cols) {
    bmm_omp_simple(a, b, c, batch_dim, a_rows, b_cols, a_cols);
}

void bmm_simd_wrapper(const double *a, const double *b, double *c,
                      int batch_dim, int a_rows, int b_cols, int a_cols) {
    bmm_simd(a, b, c, batch_dim, a_rows, b_cols, a_cols);
}

void bmm_collaps_wrapper(const double *a, const double *b, double *c,
                         int batch_dim, int a_rows, int b_cols, int a_cols) {
    bmm_collaps(a, b, c, batch_dim, a_rows, b_cols, a_cols);
}

void bmm_blocked_simd_wrapper(const double *a, const double *b, double *c,
                              int batch_dim, int a_rows, int b_cols, int a_cols, int block_size) {
    bmm_blocked_simd(a, b, c, batch_dim, a_rows, b_cols, a_cols, block_size);
}

void bmm_blocked_simd_restricted_pointers_wrapper(const double *a, const double *b, double *c,
                                                  int batch_dim, int a_rows, int b_cols, int a_cols, int block_size) {
    bmm_blocked_simd_restricted_pointers(a, b, c, batch_dim, a_rows, b_cols, a_cols, block_size);
}

bool compare_matrices(const vector<double> &a, const vector<double> &b) {
    const double epsilon = 1e-6;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

TEST_CASE("Batch Matrix Multiplication Timing and Correctness", "[bmm]") {
    const int batch_dim = 10;
    const int a_rows = 512;
    const int b_cols = 512;
    const int a_cols = 512;

    vector<double> a(batch_dim * a_rows * a_cols);
    vector<double> b(batch_dim * a_cols * b_cols);
    vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    // Reference implementation
    batch_matrix_multiply_wrapper(a.data(), b.data(), c_ref.data(), batch_dim, a_rows, b_cols, a_cols);

    SECTION("Timing and correctness of bmm_omp_simple") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_function(bmm_omp_simple_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
        cout << "Time taken by bmm_omp_simple: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }

    SECTION("Timing and correctness of bmm_simd") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_function(bmm_simd_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
        cout << "Time taken by bmm_simd: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }

    SECTION("Timing and correctness of bmm_collaps") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_function(bmm_collaps_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
        cout << "Time taken by bmm_collaps: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }

    SECTION("Timing batch_matrix_multiply") {
        double time_taken = time_function(batch_matrix_multiply_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
        cout << "Time taken by batch_matrix_multiply: " << time_taken << " seconds" << endl;
    }

    SECTION("Timing bmm_blocked with different block sizes") {
        vector<int> block_sizes = {16, 32, 64, 128};
        for (int block_size : block_sizes) {
            fill(c.begin(), c.end(), 0.0); // Reset the result matrix
            double time_taken = time_blocked_function(bmm_blocked_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, block_size);
            cout << "Time taken by bmm_blocked with block size " << block_size << ": " << time_taken << " seconds" << endl;
        }
    }
}

TEST_CASE("Comparison of bmm_blocking functions", "[bmm_blocking]") {
    const int batch_dim = 10;
    const int a_rows = 512;
    const int b_cols = 512;
    const int a_cols = 512;
    const int block_size = 32;

    vector<double> a(batch_dim * a_rows * a_cols);
    vector<double> b(batch_dim * a_cols * b_cols);
    vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    // Reference implementation
    batch_matrix_multiply_wrapper(a.data(), b.data(), c_ref.data(), batch_dim, a_rows, b_cols, a_cols);

    SECTION("Timing and correctness of bmm_blocked") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_blocked_function(bmm_blocked_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, block_size);
        cout << "Time taken by bmm_blocked: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }

    SECTION("Timing and correctness of bmm_blocked_simd") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_blocked_function(bmm_blocked_simd_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, block_size);
        cout << "Time taken by bmm_blocked_simd: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }

    SECTION("Timing and correctness of bmm_blocked_simd_restricted_pointers") {
        fill(c.begin(), c.end(), 0.0);
        double time_taken = time_blocked_function(bmm_blocked_simd_restricted_pointers_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, block_size);
        cout << "Time taken by bmm_blocked_simd_restricted_pointers: " << time_taken << " seconds" << endl;
        REQUIRE(compare_matrices(c, c_ref));
    }
}