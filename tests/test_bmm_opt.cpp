#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include "bmm.h"
#include "catch.hpp"

using namespace std;

void generate_random_matrix(vector<double> &matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

bool compare_matrices(const vector<double> &a, const vector<double> &b) {
    const double epsilon = 1e-6;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

TEST_CASE("Comparison of bmm (optimized) and batch_matrix_multiply (naive)", "[bmm_comparison]") {
    const int batch_dim = 1;
    const int a_rows = 512;
    const int b_cols = 512;
    const int a_cols = 512;

    vector<double> a(batch_dim * a_rows * a_cols);
    vector<double> b(batch_dim * a_cols * b_cols);
    vector<double> c_naive(batch_dim * a_rows * b_cols, 0.0);
    vector<double> c_optimized(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    // Naive implementation
    batch_matrix_multiply(a.data(), b.data(), c_naive.data(), batch_dim, a_rows, b_cols, a_cols);

    // Optimized implementation
    bmm(a.data(), b.data(), c_optimized.data(), batch_dim, a_rows, b_cols, a_cols);

    // Compare results
    REQUIRE(compare_matrices(c_naive, c_optimized));
}
