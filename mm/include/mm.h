//
// Created by sonja on 10.01.25.
//

#ifndef MM_H
#define MM_H

#include <immintrin.h>

void mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_transposed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_auto_vectorized(const double *a, const double *b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols);
void mm_omp_vectorized(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_32(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_64(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_pipe_2(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
//void mm_vectorized_pipe_nT(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_pipe_8(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);

// kernel_4x4 is a helper function for mm_kernel_4x4
//void kernel_4x4(const double *a, const double *b, double *c, const int i, const int j, const int k, const int k_end, const int a_rows, const int b_cols, const int a_cols);
//void mm_kernel_4x4(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
//void mm_kernel_4x4_nT(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
//void micro_kernel_4x4(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
//void mm_kernel_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void kernel(double *a, __m256d *b, __m256d *c, int x, int y, int l, int r, int n);
void mm_kernel_6x16(const double *a, const double *b, double *c, int a_rows, int b_cols, int a_cols);

#endif //MM_H


//Note to self: a_rows x a_cols * ((b_rows = a_cols)) x b_cols = a_rows x b_cols ((= c_rows x c_cols))