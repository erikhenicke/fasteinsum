#pragma once

#include <vector>
#include "aligned_allocator.h"

// aligned_vector is a 64 byte aligned std::vector
template<class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64> >;

void bmm_naive(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
               int a_cols);

void bmm_naive_parallel(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
               int a_cols);

void bmm_blas(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols);

void bmm_blas_parallel(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols);

void bmm_kernel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols, int a_cols, int h, int w,
    void (*kernel)(const double*, const double*, double*, int, int, int, int, int, int, int, int));

void bmm_kernel_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols, int a_cols, int h,
    int w, void (*kernel)(const double*, const double*, double*, int, int, int, int, int, int, int, int));

void bmm_packing(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                    int a_cols, int h, int w, int b1, int b2_, int b3_,
                    void (*kernel_pack)(const double*, const double*, double*, int, int, int, int, int, int, int, int,
                        int, int));

void bmm_packing_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                    int a_cols, int h, int w, int b1, int b2_, int b3_,
                    void (*kernel_pack)(const double*, const double*, double*, int, int, int, int, int, int, int));

/**********************************************************************************************************************
 * NOTE: The following functions are not used in the experiments.
 **********************************************************************************************************************/

void bmm_no_kernel_transpose_blocking_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                   int a_cols, int b1, int b2, int b3);

void bmm_variable_kernel_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                    int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_);

void bmm_simple_kernel_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                       int a_cols, int h, int w, int simd_length, int wl, int b1_, int b2_, int b3_);

void bmm_blocking(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
         int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(const double*, const double*, double *, int, int, int, int, int, int, int, int));

void bmm_blocking_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                  int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
                  void (*kernel)(const double*, const double*, double *, int, int, int, int, int, int,
                                 int, int));

void bmm_transpose_blocking_parallel(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                   int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
                   void (*kernel)(const double*, const double*, double *, int, int, int, int, int, int,
                                  int, int));

void bmm_blocking_parallel_inner3(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                        int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
                        void (*kernel)(const double*, const double*, double *, int, int, int, int, int,
                                       int, int, int));

void bmm_blocking_parallel_inner4(const double *a, const double *b, double *c, int bd, int a_rows, int b_cols,
                        int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
                        void (*kernel)(const double*, const double*, double *, int, int, int, int, int,
                                       int, int, int));