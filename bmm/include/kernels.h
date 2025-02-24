//
// Created by sonja on 05.02.25.
//

#ifndef KERNELS_H
#define KERNELS_H

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel_more4(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

void bmm_parallel_more5(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int));

//void kernel_omp(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl);
//
//void kernel_v2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl);

// hardcode h, w, simd_length, wl, unroll loops (pipelining)
// w = 4, 8, 12, 16, 20 (must be multiple of simd_length)
// h = 4, 6, 8, (maybe later 10, 12, 16, ...)
// simd_length = 4
// wl = w / simd_length

void kernel_2x24(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x12_test1(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x12_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_4x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_6x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_6x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_6x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_6x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_6x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x16_test(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x16_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);

void kernel_8x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
//             int a_idx, int b_idx, int l, int r);

#endif //KERNELS_H
