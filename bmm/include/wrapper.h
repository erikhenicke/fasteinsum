#pragma once
#include <vector>
#include "aligned_allocator.h"

template<class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64> >;

void bmm_naive_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                       const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel8x16_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                            const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel4x12_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                            const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_simple_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                               const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

// no vector intrinsics, transferable to other architectures
void bmm_blocked_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                         const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_blas_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                      const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V1_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V2_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V3_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                        const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_T_V4_wrapper(const double *a, const double *b, double *c, const int batch_dim, const int a_rows,
                      const int b_cols, const int a_cols, int b1, int b2, int b3, double *time);

void bmm_pack(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols,
              const int a_cols, int b1, int b2, int b3, double *time);

// verschiedene Parallelisie
