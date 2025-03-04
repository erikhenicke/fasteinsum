//
// Created by sonja on 05.02.25.
//

#ifndef KERNELS_H
#define KERNELS_H


//void kernel_omp(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl);
//
//void kernel_v2(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl);

// hardcode h, w, simd_length, wl, unroll loops (pipelining)
// w = 4, 8, 12, 16, 20 (must be multiple of simd_length)
// h = 4, 6, 8, (maybe later 10, 12, 16, ...)
// simd_length = 4
// wl = w / simd_length
void kernel_omp_8x16_v1(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                        int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_omp_8x16_v2(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                        int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_omp_8x16_v3(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                        int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_T_v4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel2(double *a_aligned, double *b_aligned, double *c, int d, int a_rows, int b_cols,
             int a_cols, int a_idx, int b_idx, int l, int r, int height, int width);

void kernel_var(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r, int h, int w, int simd_length,
                int wl);

void kernel_2x24(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x12_test1(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                       int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x12_test2(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                       int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_4x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_6x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_6x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_6x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_6x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_6x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x16_test(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                      int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x16_test2(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                       int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows,
                 int b_cols, int a_cols, int a_idx, int b_idx, int l, int r);

void kernel_8x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, int c_cols, int b_pack_cols,
                 int a_idx, int a_pack_idx, int b_pack_idx, int l, int r);

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

//
//void kernel_10x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_10x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_12x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x4(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x8(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x12(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x16(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);
//
//void kernel_16x20(double *a_aligned, double *b_aligned, double *c_aligned, int d, int a_rows, int b_cols, int a_cols,//             int a_idx, int b_idx, int l, int r);

#endif //KERNELS_H
