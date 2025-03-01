#pragma once

#include <vector>
#include "aligned_allocator.h"

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;


void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned, aligned_vector<double> &c_aligned,
          const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);

void packT(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned, aligned_vector<double> &c_aligned,
          const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);

void pack_T_pad(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned, aligned_vector<double> &c_aligned,
           const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded);


void bmm_naive(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

void bmm_var_kernel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_);

void bmm_simple_kernel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1_, int b2_, int b3_,
    void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int, int, int));


void bmm_T_bl_para(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int b1, int b2, int b3);

void bmm_blas(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallelT(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel_more4(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel_more5(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));
