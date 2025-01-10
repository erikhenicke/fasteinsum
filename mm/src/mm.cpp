//
// Created by sonja on 10.01.25.
//

#include "mm.h"
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Naive implementation using three nested loops
void *mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];
}

// Transpose matrix b so that is is in column-major order before multiplication
// Then column of b needed in each iteration is contiguous in memory -> better cache utilization
void *mm_transposed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
  // Transpose matrix b
  double *b_transposed = new double[a_cols*b_cols];
    for (int i = 0; i < a_cols; ++i)
        for (int j = 0; j < b_cols; ++j)
            b_transposed[i*b_cols + j] = b[j*a_cols + i];
    // Naive MM (Note different indexing of b!)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b_transposed[j*a_cols + k];
    // Clean up
    delete[] b_transposed;
}

// Vectorization with SIMD instructions (automatically by compiler with __restrict__ keyword and manually)
void *mm_auto_vectorized(const double *a, const double *b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols) {
    // Transpose matrix b
    double *b_transposed = new double[a_cols*b_cols];
    for (int i = 0; i < a_cols; ++i)
        for (int j = 0; j < b_cols; ++j)
            b_transposed[i*b_cols + j] = b[j*a_cols + i];
    // Naive MM (Note different indexing of b!)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b_transposed[j*a_cols + k];
    // Clean up
    delete[] b_transposed;
}

// align arrays to 64 bit cache line size
void align_and_transpose_matrices(const double *a, const double *b, double **aligned_a, double **aligned_b, int a_rows, int a_cols, int b_cols) {
    // Allocate aligned memory for matrices A and B
    posix_memalign(reinterpret_cast<void**>(aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(*aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            (*aligned_b)[i * b_cols + j] = b[j * a_cols + i];
        }
    }
}

void *mm_omp_vectorized(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    posix_memalign(reinterpret_cast<void**>(aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(*aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            (*aligned_b)[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    #pragma omp simd aligned(a, b, c: 64)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += aligned_a[i*a_cols + k] * aligned_b[j*a_cols + k];
}

void *mm_vectorized(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    posix_memalign(reinterpret_cast<void**>(aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(*aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            (*aligned_b)[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Use Vector instructions to compute matrix multiplication