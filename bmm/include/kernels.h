//
// Created by sonja on 05.02.25.
//

#ifndef KERNELS_H
#define KERNELS_H

// hardcode h, w, simd_length, wl, unroll loops (pipelining)
// w must be multiple of simd_length
// simd_length = 4
// wl = w / simd_length

void kernel_8x16(const double *a_aligned, const double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, int d, int a_rows,
                      int b_cols, int b_pack_cols, int a_idx, int b_idx, int a_pack_idx,
                      int b_pack_idx, int l, int r);

void kernel_8x16_pack_omp(const double *a_packed, const double *b_packed, double *c_aligned, int d, int a_rows,
                          int b_cols, int b_pack_cols, int a_idx, int b_idx, int a_pack_idx, int b_pack_idx, int l,
                          int r);

void kernel_8x16_pack_omp_unrolled(const double *a_packed, const double *b_packed, double *c_aligned, int d, int a_rows,
                          int b_cols, int b_pack_cols, int a_idx, int b_idx, int a_pack_idx, int b_pack_idx, int l,
                          int r);

void kernel_8x16_pack_local_c(const double *a_packed, const double *b_packed, double *c_local, int c_cols, int b_pack_cols,
                              int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_8x16_pack_local_c_omp(const double *a_packed, const double *b_packed, double *c_local, int c_cols,
                                  int b_pack_cols, int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_8x16_pack_local_c_omp_unrolled(const double *a_packed, const double *b_packed, double *c_local, int c_cols,
                                  int b_pack_cols, int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_14x8_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_4x12_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_10x12_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_14x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_18x20_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_12x24_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

void kernel_12x32_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

#endif //KERNELS_H
