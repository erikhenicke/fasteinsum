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

template<class T>
using aligned_vector = vector<T, aligned_allocator<T, 64> >;

using namespace std;

constexpr int SIMD_LENGTH = 4;

// Wrapper function signature
// void (*bmm)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, double*);,

void bmm_naive_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                       const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_naive(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_naive_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                       const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_naive_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_blas_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                      const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blas(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_blas_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                      const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blas_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_kernel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, kernel_8x16);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_kernel_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, kernel_8x16);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_packing_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_packing_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack_local_c);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_packing_omp_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack_omp);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_packing_omp_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack_local_c_omp);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

/**********************************************************************************************************************
 * NOTE: The following functions are not used in the experiments.
 **********************************************************************************************************************/

void bmm_packing_omp_unrolled_parallel_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack_local_c_omp_unrolled);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_packing_omp_unrolled_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, b1, b2, b3, kernel_8x16_pack_omp_unrolled);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel8x16_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                            const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_8x16);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel4x12_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                            const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 4, 12, SIMD_LENGTH, 1, b1, b2, b3, kernel_4x12);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_omp_V1_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_omp_8x16_v1);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_omp_V2_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_omp_8x16_v2);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_omp_V3_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_omp_8x16_v3);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_T_V4_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                      const int b_cols, const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_transpose_blocking_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 8, 16, SIMD_LENGTH, 2, b1, b2, b3, kernel_T_v4);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_14x8_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 14, 8, b1, b2, b3, kernel_14x8_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_4x12_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 4, 12, b1, b2, b3, kernel_4x12_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_10x12_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 10, 12, b1, b2, b3, kernel_10x12_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_14x16_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 14, 16, b1, b2, b3, kernel_14x16_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_18x20_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 18, 20, b1, b2, b3, kernel_18x20_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_12x24_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 12, 24, b1, b2, b3, kernel_12x24_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}

void bmm_kernel_12x32_pack_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time) {
    auto start = high_resolution_clock::now();
    bmm_packing_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols, 12, 32, b1, b2, b3, kernel_12x32_pack);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    *time = elapsed.count();
}





