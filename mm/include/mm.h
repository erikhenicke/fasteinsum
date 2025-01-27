//
// Created by sonja on 10.01.25.
//

#ifndef MM_H
#define MM_H

#include <vector>
#include "aligned_allocator.h"

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

// Matrix Multiplication Functions
void mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_transposed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_auto_vectorized(const double *a, const double *b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols);
void mm_omp_vectorized(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols);
//void mm_vectorized_32(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_64(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_pipe_2(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
//void mm_vectorized_pipe_nT(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_vectorized_pipe_8(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);

void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_kernel2(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_blocked(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_blocked_packed_stdmin(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);
void mm_blocked_packed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);


// Helper Functions
void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned_transposed, const int a_rows, const int a_cols, const int b_cols);
void kernel(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int height, int width, int l, int r) ;
void kernel2(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int height, int width, int l, int r) ;

#endif //MM_H


//Note to self: a_rows x a_cols * ((b_rows = a_cols)) x b_cols = a_rows x b_cols ((= c_rows x c_cols))