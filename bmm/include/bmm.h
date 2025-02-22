#pragma once


#include <vector>
#include "aligned_allocator.h"

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

double *bmm_naive(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_);

void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) ;

void generate_random_matrix(aligned_vector<double> &matrix, int rows, int cols);

void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned, aligned_vector<double> &c_aligned,
          const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);

void bmm2(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int b1, int b2_, int b3_);

void kernel2(double *a_aligned, double *b_aligned, double *c, const int d, const int a_rows, const int b_cols, const int a_cols, int a_idx, int b_idx, int l, int r, int height, int width);

void pack2(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned, aligned_vector<double> &c_aligned,
           const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);