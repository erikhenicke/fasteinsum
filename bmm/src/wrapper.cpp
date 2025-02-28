#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>

#include "aligned_allocator.h"
#include "kernels.h"
#include "bmm.h"

using namespace std;
using namespace std::chrono;

template <class T>
using aligned_vector = vector<T, aligned_allocator<T, 64>>;

using namespace std;

constexpr int SIMD_LENGTH = 4;

// wrapper function signature
// void (*bmm)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, double*);,

void bmm_naive_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_naive(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel8x16_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_8x16);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel4x12_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 4, 12, SIMD_LENGTH, 1, b1, b2, b3, kernel_4x12);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_simple_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_simple_kernel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 6, 8, SIMD_LENGTH, 2, b1, b2, b3, kernel2);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_blocked_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_T_bl_para(a, b, c, batch_dim, a_rows, b_cols, a_cols, b1, b2, b3);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_blas_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3, double * time) {
    auto start = high_resolution_clock::now();
    bmm_blas(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}