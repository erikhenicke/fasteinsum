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

/**********************************************************************************************************************
 * NOTE: The following functions are not used in the experiments.
 **********************************************************************************************************************/

void bmm_packing_omp_unrolled_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_packing_omp_unrolled_parallel_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel8x16_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                            int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel4x12_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                            int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_simple_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                               int b_cols, int a_cols, int b1, int b2, int b3, double *time);

// no vector intrinsics, transferable to other architectures
void bmm_blocked_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                         int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V1_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V2_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_omp_V3_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                        int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_T_V4_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows,
                      int b_cols, int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_14x8_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_4x12_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_10x12_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_14x16_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_18x20_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_12x24_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

void bmm_kernel_12x32_pack_wrapper(const double *a, const double *b, double *c, int batch_dim, int a_rows, int b_cols,
              int a_cols, int b1, int b2, int b3, double *time);

// verschiedene Parallelisie
