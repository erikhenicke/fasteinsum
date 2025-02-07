#pragma once
#include <vector>
#include "aligned_allocator.h"

template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);

double *bmm_naive(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_);