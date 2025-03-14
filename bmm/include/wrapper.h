#pragma once
#include <vector>
#include "aligned_allocator.h"

template<class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64> >;

void bmm_naive_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                       int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_naive_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                       int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_blas_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                      int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_blas_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                      int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_packing_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_packing_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_packing_omp_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_packing_omp_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);