#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <functional>
#include <omp.h>

// aligned_vector is a 64 byte aligned std::vector
template<class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64> >;

using namespace std;

void kernel_8x16(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
                 int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[offsetB + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[offsetA + 4 * a_cols]);
        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[offsetA + 5 * a_cols]);
        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[offsetA + 6 * a_cols]);
        t[24] = _mm256_fmadd_pd(a6, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a6, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a6, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a6, b3, t[27]);

        __m256d a7 = _mm256_broadcast_sd(&a_aligned[offsetA + 7 * a_cols]);
        t[28] = _mm256_fmadd_pd(a7, b0, t[28]);
        t[29] = _mm256_fmadd_pd(a7, b1, t[29]);
        t[30] = _mm256_fmadd_pd(a7, b2, t[30]);
        t[31] = _mm256_fmadd_pd(a7, b3, t[31]);

        offsetA++;
        offsetB += b_cols;
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_8x16_pack_local_c(const double *a_packed, const double *b_packed, double *c_local, const int c_cols,
                              const int b_pack_cols, const int a_idx, const int a_pack_idx,
                              const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(32);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + 1 * a_pack_cols]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a6, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a6, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a6, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a6, b3, t[27]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[28] = _mm256_fmadd_pd(a7, b0, t[28]);
        t[29] = _mm256_fmadd_pd(a7, b1, t[29]);
        t[30] = _mm256_fmadd_pd(a7, b2, t[30]);
        t[31] = _mm256_fmadd_pd(a7, b3, t[31]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_local[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_8x16_pack_local_c_omp(const double *a_packed, const double *b_packed, double *c_local, const int c_cols,
                                  const int b_pack_cols,
                                  const int a_idx, const int a_pack_idx, const int b_pack_idx, const int l,
                                  const int r) {
    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;

    // Temporary storage for the results
    double t[8][16] = {0};

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            t[i][j] = 0.0;
        }
    }

    for (int k = l; k < r; k++) {
        for (int i = 0; i < 8; ++i) {
#pragma omp simd
            for (int j = 0; j < 16; ++j) {
                t[i][j] += a_packed[offsetA + i * a_pack_cols] * b_packed[offsetB + j];
            }
        }
        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            c_local[(a_idx + i) * c_cols + b_pack_idx + j] += t[i][j];
        }
    }
}

void kernel_8x16_pack_local_c_omp_unrolled(const double *a_packed, const double *b_packed, double *c_local, const int c_cols,
                                  const int b_pack_cols,
                                  const int a_idx, const int a_pack_idx, const int b_pack_idx, const int l,
                                  const int r) {
    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;

    // Temporary storage for the results
    double t[8 * 16] = {0};

    // Unrolled initialization loop
    #pragma omp simd
    for (int j = 0; j < 16; ++j) {
        t[0 * 16 + j] = 0.0;
        t[1 * 16 + j] = 0.0;
        t[2 * 16 + j] = 0.0;
        t[3 * 16 + j] = 0.0;
        t[4 * 16 + j] = 0.0;
        t[5 * 16 + j] = 0.0;
        t[6 * 16 + j] = 0.0;
        t[7 * 16 + j] = 0.0;
    }

    for (int k = l; k < r; k++) {
        // Unrolled computation loop
        #pragma omp simd
        for (int j = 0; j < 16; ++j) {
            t[0 * 16 + j] += a_packed[offsetA + 0 * a_pack_cols] * b_packed[offsetB + j];
            t[1 * 16 + j] += a_packed[offsetA + 1 * a_pack_cols] * b_packed[offsetB + j];
            t[2 * 16 + j] += a_packed[offsetA + 2 * a_pack_cols] * b_packed[offsetB + j];
            t[3 * 16 + j] += a_packed[offsetA + 3 * a_pack_cols] * b_packed[offsetB + j];
            t[4 * 16 + j] += a_packed[offsetA + 4 * a_pack_cols] * b_packed[offsetB + j];
            t[5 * 16 + j] += a_packed[offsetA + 5 * a_pack_cols] * b_packed[offsetB + j];
            t[6 * 16 + j] += a_packed[offsetA + 6 * a_pack_cols] * b_packed[offsetB + j];
            t[7 * 16 + j] += a_packed[offsetA + 7 * a_pack_cols] * b_packed[offsetB + j];
        }
        offsetA++;
        offsetB += b_pack_cols;
    }

    int offsetC = a_idx * c_cols + b_pack_idx;
    // Unrolled final accumulation loop
    #pragma omp simd
    for (int j = 0; j < 16; ++j) {
        c_local[offsetC + 0 * c_cols + j] += t[0 * 16 + j];
        c_local[offsetC + 1 * c_cols + j] += t[1 * 16 + j];
        c_local[offsetC + 2 * c_cols + j] += t[2 * 16 + j];
        c_local[offsetC + 3 * c_cols + j] += t[3 * 16 + j];
        c_local[offsetC + 4 * c_cols + j] += t[4 * 16 + j];
        c_local[offsetC + 5 * c_cols + j] += t[5 * 16 + j];
        c_local[offsetC + 6 * c_cols + j] += t[6 * 16 + j];
        c_local[offsetC + 7 * c_cols + j] += t[7 * 16 + j];
    }
}

void kernel_8x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int d, const int a_rows,
                      const int b_cols, const int b_pack_cols, const int a_idx, const int b_idx, const int a_pack_idx,
                      const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(32);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + 1 * a_pack_cols]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a6, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a6, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a6, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a6, b3, t[27]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[28] = _mm256_fmadd_pd(a7, b0, t[28]);
        t[29] = _mm256_fmadd_pd(a7, b1, t[29]);
        t[30] = _mm256_fmadd_pd(a7, b2, t[30]);
        t[31] = _mm256_fmadd_pd(a7, b3, t[31]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_8x16_pack_omp(const double *a_packed, const double *b_packed, double *c_aligned, const int d,
                          const int a_rows,
                          const int b_cols, const int b_pack_cols, const int a_idx, const int b_idx,
                          const int a_pack_idx,
                          const int b_pack_idx, const int l, const int r) {
    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;

    // Temporary storage for the results
    double t[8][16] = {0};

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            t[i][j] = 0.0;
        }
    }

    for (int k = l; k < r; k++) {
        for (int i = 0; i < 8; ++i) {
#pragma omp simd
            for (int j = 0; j < 16; ++j) {
                t[i][j] += a_packed[offsetA + i * a_pack_cols] * b_packed[offsetB + j];
            }
        }
        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j] += t[i][j];
        }
    }
}

void kernel_8x16_pack_omp_unrolled(const double *a_packed, const double *b_packed, double *c_aligned, const int d,
                          const int a_rows,
                          const int b_cols, const int b_pack_cols, const int a_idx, const int b_idx,
                          const int a_pack_idx,
                          const int b_pack_idx, const int l, const int r) {
    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;

    // Temporary storage for the results
    double t[8 * 16] = {0};

    // Unrolled initialization loop
    #pragma omp simd
    for (int j = 0; j < 16; ++j) {
        t[0 * 16 + j] = 0.0;
        t[1 * 16 + j] = 0.0;
        t[2 * 16 + j] = 0.0;
        t[3 * 16 + j] = 0.0;
        t[4 * 16 + j] = 0.0;
        t[5 * 16 + j] = 0.0;
        t[6 * 16 + j] = 0.0;
        t[7 * 16 + j] = 0.0;
    }

    for (int k = l; k < r; k++) {
        // Unrolled computation loop
        #pragma omp simd
        for (int j = 0; j < 16; ++j) {
            t[0 * 16 + j] += a_packed[offsetA + 0 * a_pack_cols] * b_packed[offsetB + j];
            t[1 * 16 + j] += a_packed[offsetA + 1 * a_pack_cols] * b_packed[offsetB + j];
            t[2 * 16 + j] += a_packed[offsetA + 2 * a_pack_cols] * b_packed[offsetB + j];
            t[3 * 16 + j] += a_packed[offsetA + 3 * a_pack_cols] * b_packed[offsetB + j];
            t[4 * 16 + j] += a_packed[offsetA + 4 * a_pack_cols] * b_packed[offsetB + j];
            t[5 * 16 + j] += a_packed[offsetA + 5 * a_pack_cols] * b_packed[offsetB + j];
            t[6 * 16 + j] += a_packed[offsetA + 6 * a_pack_cols] * b_packed[offsetB + j];
            t[7 * 16 + j] += a_packed[offsetA + 7 * a_pack_cols] * b_packed[offsetB + j];
        }
        offsetA++;
        offsetB += b_pack_cols;
    }

    int offsetC = (d * a_rows + a_idx) * b_cols + b_idx;
    // Unrolled final accumulation loop
    #pragma omp simd
    for (int j = 0; j < 16; ++j) {
        c_aligned[offsetC + 0 * b_cols + j] += t[0 * 16 + j];
        c_aligned[offsetC + 1 * b_cols + j] += t[1 * 16 + j];
        c_aligned[offsetC + 2 * b_cols + j] += t[2 * 16 + j];
        c_aligned[offsetC + 3 * b_cols + j] += t[3 * 16 + j];
        c_aligned[offsetC + 4 * b_cols + j] += t[4 * 16 + j];
        c_aligned[offsetC + 5 * b_cols + j] += t[5 * 16 + j];
        c_aligned[offsetC + 6 * b_cols + j] += t[6 * 16 + j];
        c_aligned[offsetC + 7 * b_cols + j] += t[7 * 16 + j];
    }
}


void kernel_14x8_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                      const int b_pack_cols, const int a_idx, const int a_pack_idx,
                      const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(28);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 28; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + 1 * a_pack_cols]);
        t[2] = _mm256_fmadd_pd(a1, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a1, b1, t[3]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[4] = _mm256_fmadd_pd(a2, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a2, b1, t[5]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[6] = _mm256_fmadd_pd(a3, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[8] = _mm256_fmadd_pd(a4, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a4, b1, t[9]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[10] = _mm256_fmadd_pd(a5, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a5, b1, t[11]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a6, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a6, b1, t[13]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[14] = _mm256_fmadd_pd(a7, b0, t[14]);
        t[15] = _mm256_fmadd_pd(a7, b1, t[15]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[16] = _mm256_fmadd_pd(a8, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a8, b1, t[17]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[18] = _mm256_fmadd_pd(a9, b0, t[18]);
        t[19] = _mm256_fmadd_pd(a9, b1, t[19]);

        __m256d a10 = _mm256_broadcast_sd(&a_packed[offsetA + 10 * a_pack_cols]);
        t[20] = _mm256_fmadd_pd(a10, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a10, b1, t[21]);

        __m256d a11 = _mm256_broadcast_sd(&a_packed[offsetA + 11 * a_pack_cols]);
        t[22] = _mm256_fmadd_pd(a11, b0, t[22]);
        t[23] = _mm256_fmadd_pd(a11, b1, t[23]);

        __m256d a12 = _mm256_broadcast_sd(&a_packed[offsetA + 12 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a12, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a12, b1, t[25]);

        __m256d a13 = _mm256_broadcast_sd(&a_packed[offsetA + 13 * a_pack_cols]);
        t[26] = _mm256_fmadd_pd(a13, b0, t[26]);
        t[27] = _mm256_fmadd_pd(a13, b1, t[27]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 2 + j][k];
            }
        }
    }
}

void kernel_4x12_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                      const int b_pack_cols, const int a_idx, const int a_pack_idx,
                      const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(12);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + 1 * a_pack_cols]);
        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_10x12_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                       const int b_pack_cols, const int a_idx, const int a_pack_idx,
                       const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(30);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 30; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + a_pack_cols]);
        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a4, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a4, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a4, b2, t[14]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[15] = _mm256_fmadd_pd(a5, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a5, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a5, b2, t[17]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[18] = _mm256_fmadd_pd(a6, b0, t[18]);
        t[19] = _mm256_fmadd_pd(a6, b1, t[19]);
        t[20] = _mm256_fmadd_pd(a6, b2, t[20]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[21] = _mm256_fmadd_pd(a7, b0, t[21]);
        t[22] = _mm256_fmadd_pd(a7, b1, t[22]);
        t[23] = _mm256_fmadd_pd(a7, b2, t[23]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a8, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a8, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a8, b2, t[26]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[27] = _mm256_fmadd_pd(a9, b0, t[27]);
        t[28] = _mm256_fmadd_pd(a9, b1, t[28]);
        t[29] = _mm256_fmadd_pd(a9, b2, t[29]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_14x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                       const int b_pack_cols, const int a_idx, const int a_pack_idx,
                       const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(56);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 56; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + a_pack_cols]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a6, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a6, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a6, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a6, b3, t[27]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[28] = _mm256_fmadd_pd(a7, b0, t[28]);
        t[29] = _mm256_fmadd_pd(a7, b1, t[29]);
        t[30] = _mm256_fmadd_pd(a7, b2, t[30]);
        t[31] = _mm256_fmadd_pd(a7, b3, t[31]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[32] = _mm256_fmadd_pd(a8, b0, t[32]);
        t[33] = _mm256_fmadd_pd(a8, b1, t[33]);
        t[34] = _mm256_fmadd_pd(a8, b2, t[34]);
        t[35] = _mm256_fmadd_pd(a8, b3, t[35]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[36] = _mm256_fmadd_pd(a9, b0, t[36]);
        t[37] = _mm256_fmadd_pd(a9, b1, t[37]);
        t[38] = _mm256_fmadd_pd(a9, b2, t[38]);
        t[39] = _mm256_fmadd_pd(a9, b3, t[39]);

        __m256d a10 = _mm256_broadcast_sd(&a_packed[offsetA + 10 * a_pack_cols]);
        t[40] = _mm256_fmadd_pd(a10, b0, t[40]);
        t[41] = _mm256_fmadd_pd(a10, b1, t[41]);
        t[42] = _mm256_fmadd_pd(a10, b2, t[42]);
        t[43] = _mm256_fmadd_pd(a10, b3, t[43]);

        __m256d a11 = _mm256_broadcast_sd(&a_packed[offsetA + 11 * a_pack_cols]);
        t[44] = _mm256_fmadd_pd(a11, b0, t[44]);
        t[45] = _mm256_fmadd_pd(a11, b1, t[45]);
        t[46] = _mm256_fmadd_pd(a11, b2, t[46]);
        t[47] = _mm256_fmadd_pd(a11, b3, t[47]);

        __m256d a12 = _mm256_broadcast_sd(&a_packed[offsetA + 12 * a_pack_cols]);
        t[48] = _mm256_fmadd_pd(a12, b0, t[48]);
        t[49] = _mm256_fmadd_pd(a12, b1, t[49]);
        t[50] = _mm256_fmadd_pd(a12, b2, t[50]);
        t[51] = _mm256_fmadd_pd(a12, b3, t[51]);

        __m256d a13 = _mm256_broadcast_sd(&a_packed[offsetA + 13 * a_pack_cols]);
        t[52] = _mm256_fmadd_pd(a13, b0, t[52]);
        t[53] = _mm256_fmadd_pd(a13, b1, t[53]);
        t[54] = _mm256_fmadd_pd(a13, b2, t[54]);
        t[55] = _mm256_fmadd_pd(a13, b3, t[55]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_18x20_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                       const int b_pack_cols, const int a_idx, const int a_pack_idx,
                       const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(90);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 90; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);
        __m256d b4 = _mm256_load_pd(&b_packed[offsetB + 16]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + a_pack_cols]);
        t[5] = _mm256_fmadd_pd(a1, b0, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b1, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b2, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b3, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b4, t[9]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[10] = _mm256_fmadd_pd(a2, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b1, t[11]);
        t[12] = _mm256_fmadd_pd(a2, b2, t[12]);
        t[13] = _mm256_fmadd_pd(a2, b3, t[13]);
        t[14] = _mm256_fmadd_pd(a2, b4, t[14]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[15] = _mm256_fmadd_pd(a3, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a3, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a3, b2, t[17]);
        t[18] = _mm256_fmadd_pd(a3, b3, t[18]);
        t[19] = _mm256_fmadd_pd(a3, b4, t[19]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[20] = _mm256_fmadd_pd(a4, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a4, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a4, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a4, b3, t[23]);
        t[24] = _mm256_fmadd_pd(a4, b4, t[24]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[25] = _mm256_fmadd_pd(a5, b0, t[25]);
        t[26] = _mm256_fmadd_pd(a5, b1, t[26]);
        t[27] = _mm256_fmadd_pd(a5, b2, t[27]);
        t[28] = _mm256_fmadd_pd(a5, b3, t[28]);
        t[29] = _mm256_fmadd_pd(a5, b4, t[29]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[30] = _mm256_fmadd_pd(a6, b0, t[30]);
        t[31] = _mm256_fmadd_pd(a6, b1, t[31]);
        t[32] = _mm256_fmadd_pd(a6, b2, t[32]);
        t[33] = _mm256_fmadd_pd(a6, b3, t[33]);
        t[34] = _mm256_fmadd_pd(a6, b4, t[34]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[35] = _mm256_fmadd_pd(a7, b0, t[35]);
        t[36] = _mm256_fmadd_pd(a7, b1, t[36]);
        t[37] = _mm256_fmadd_pd(a7, b2, t[37]);
        t[38] = _mm256_fmadd_pd(a7, b3, t[38]);
        t[39] = _mm256_fmadd_pd(a7, b4, t[39]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[40] = _mm256_fmadd_pd(a8, b0, t[40]);
        t[41] = _mm256_fmadd_pd(a8, b1, t[41]);
        t[42] = _mm256_fmadd_pd(a8, b2, t[42]);
        t[43] = _mm256_fmadd_pd(a8, b3, t[43]);
        t[44] = _mm256_fmadd_pd(a8, b4, t[44]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[45] = _mm256_fmadd_pd(a9, b0, t[45]);
        t[46] = _mm256_fmadd_pd(a9, b1, t[46]);
        t[47] = _mm256_fmadd_pd(a9, b2, t[47]);
        t[48] = _mm256_fmadd_pd(a9, b3, t[48]);
        t[49] = _mm256_fmadd_pd(a9, b4, t[49]);

        __m256d a10 = _mm256_broadcast_sd(&a_packed[offsetA + 10 * a_pack_cols]);
        t[50] = _mm256_fmadd_pd(a10, b0, t[50]);
        t[51] = _mm256_fmadd_pd(a10, b1, t[51]);
        t[52] = _mm256_fmadd_pd(a10, b2, t[52]);
        t[53] = _mm256_fmadd_pd(a10, b3, t[53]);
        t[54] = _mm256_fmadd_pd(a10, b4, t[54]);

        __m256d a11 = _mm256_broadcast_sd(&a_packed[offsetA + 11 * a_pack_cols]);
        t[55] = _mm256_fmadd_pd(a11, b0, t[55]);
        t[56] = _mm256_fmadd_pd(a11, b1, t[56]);
        t[57] = _mm256_fmadd_pd(a11, b2, t[57]);
        t[58] = _mm256_fmadd_pd(a11, b3, t[58]);
        t[59] = _mm256_fmadd_pd(a11, b4, t[59]);

        __m256d a12 = _mm256_broadcast_sd(&a_packed[offsetA + 12 * a_pack_cols]);
        t[60] = _mm256_fmadd_pd(a12, b0, t[60]);
        t[61] = _mm256_fmadd_pd(a12, b1, t[61]);
        t[62] = _mm256_fmadd_pd(a12, b2, t[62]);
        t[63] = _mm256_fmadd_pd(a12, b3, t[63]);
        t[64] = _mm256_fmadd_pd(a12, b4, t[64]);

        __m256d a13 = _mm256_broadcast_sd(&a_packed[offsetA + 13 * a_pack_cols]);
        t[65] = _mm256_fmadd_pd(a13, b0, t[65]);
        t[66] = _mm256_fmadd_pd(a13, b1, t[66]);
        t[67] = _mm256_fmadd_pd(a13, b2, t[67]);
        t[68] = _mm256_fmadd_pd(a13, b3, t[68]);
        t[69] = _mm256_fmadd_pd(a13, b4, t[69]);

        __m256d a14 = _mm256_broadcast_sd(&a_packed[offsetA + 14 * a_pack_cols]);
        t[70] = _mm256_fmadd_pd(a14, b0, t[70]);
        t[71] = _mm256_fmadd_pd(a14, b1, t[71]);
        t[72] = _mm256_fmadd_pd(a14, b2, t[72]);
        t[73] = _mm256_fmadd_pd(a14, b3, t[73]);
        t[74] = _mm256_fmadd_pd(a14, b4, t[74]);

        __m256d a15 = _mm256_broadcast_sd(&a_packed[offsetA + 15 * a_pack_cols]);
        t[75] = _mm256_fmadd_pd(a15, b0, t[75]);
        t[76] = _mm256_fmadd_pd(a15, b1, t[76]);
        t[77] = _mm256_fmadd_pd(a15, b2, t[77]);
        t[78] = _mm256_fmadd_pd(a15, b3, t[78]);
        t[79] = _mm256_fmadd_pd(a15, b4, t[79]);

        __m256d a16 = _mm256_broadcast_sd(&a_packed[offsetA + 16 * a_pack_cols]);
        t[80] = _mm256_fmadd_pd(a16, b0, t[80]);
        t[81] = _mm256_fmadd_pd(a16, b1, t[81]);
        t[82] = _mm256_fmadd_pd(a16, b2, t[82]);
        t[83] = _mm256_fmadd_pd(a16, b3, t[83]);
        t[84] = _mm256_fmadd_pd(a16, b4, t[84]);

        __m256d a17 = _mm256_broadcast_sd(&a_packed[offsetA + 17 * a_pack_cols]);
        t[85] = _mm256_fmadd_pd(a17, b0, t[85]);
        t[86] = _mm256_fmadd_pd(a17, b1, t[86]);
        t[87] = _mm256_fmadd_pd(a17, b2, t[87]);
        t[88] = _mm256_fmadd_pd(a17, b3, t[88]);
        t[89] = _mm256_fmadd_pd(a17, b4, t[89]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 18; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 5 + j][k];
            }
        }
    }
}

void kernel_12x24_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                       const int b_pack_cols, const int a_idx, const int a_pack_idx,
                       const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(72);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 72; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);
        __m256d b4 = _mm256_load_pd(&b_packed[offsetB + 16]);
        __m256d b5 = _mm256_load_pd(&b_packed[offsetB + 20]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);
        t[5] = _mm256_fmadd_pd(a0, b5, t[5]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + a_pack_cols]);
        t[6] = _mm256_fmadd_pd(a1, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b2, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b3, t[9]);
        t[10] = _mm256_fmadd_pd(a1, b4, t[10]);
        t[11] = _mm256_fmadd_pd(a1, b5, t[11]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[12] = _mm256_fmadd_pd(a2, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a2, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a2, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a2, b3, t[15]);
        t[16] = _mm256_fmadd_pd(a2, b4, t[16]);
        t[17] = _mm256_fmadd_pd(a2, b5, t[17]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[18] = _mm256_fmadd_pd(a3, b0, t[18]);
        t[19] = _mm256_fmadd_pd(a3, b1, t[19]);
        t[20] = _mm256_fmadd_pd(a3, b2, t[20]);
        t[21] = _mm256_fmadd_pd(a3, b3, t[21]);
        t[22] = _mm256_fmadd_pd(a3, b4, t[22]);
        t[23] = _mm256_fmadd_pd(a3, b5, t[23]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a4, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a4, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a4, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a4, b3, t[27]);
        t[28] = _mm256_fmadd_pd(a4, b4, t[28]);
        t[29] = _mm256_fmadd_pd(a4, b5, t[29]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[30] = _mm256_fmadd_pd(a5, b0, t[30]);
        t[31] = _mm256_fmadd_pd(a5, b1, t[31]);
        t[32] = _mm256_fmadd_pd(a5, b2, t[32]);
        t[33] = _mm256_fmadd_pd(a5, b3, t[33]);
        t[34] = _mm256_fmadd_pd(a5, b4, t[34]);
        t[35] = _mm256_fmadd_pd(a5, b5, t[35]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[36] = _mm256_fmadd_pd(a6, b0, t[36]);
        t[37] = _mm256_fmadd_pd(a6, b1, t[37]);
        t[38] = _mm256_fmadd_pd(a6, b2, t[38]);
        t[39] = _mm256_fmadd_pd(a6, b3, t[39]);
        t[40] = _mm256_fmadd_pd(a6, b4, t[40]);
        t[41] = _mm256_fmadd_pd(a6, b5, t[41]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[42] = _mm256_fmadd_pd(a7, b0, t[42]);
        t[43] = _mm256_fmadd_pd(a7, b1, t[43]);
        t[44] = _mm256_fmadd_pd(a7, b2, t[44]);
        t[45] = _mm256_fmadd_pd(a7, b3, t[45]);
        t[46] = _mm256_fmadd_pd(a7, b4, t[46]);
        t[47] = _mm256_fmadd_pd(a7, b5, t[47]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[48] = _mm256_fmadd_pd(a8, b0, t[48]);
        t[49] = _mm256_fmadd_pd(a8, b1, t[49]);
        t[50] = _mm256_fmadd_pd(a8, b2, t[50]);
        t[51] = _mm256_fmadd_pd(a8, b3, t[51]);
        t[52] = _mm256_fmadd_pd(a8, b4, t[52]);
        t[53] = _mm256_fmadd_pd(a8, b5, t[53]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[54] = _mm256_fmadd_pd(a9, b0, t[54]);
        t[55] = _mm256_fmadd_pd(a9, b1, t[55]);
        t[56] = _mm256_fmadd_pd(a9, b2, t[56]);
        t[57] = _mm256_fmadd_pd(a9, b3, t[57]);
        t[58] = _mm256_fmadd_pd(a9, b4, t[58]);
        t[59] = _mm256_fmadd_pd(a9, b5, t[59]);

        __m256d a10 = _mm256_broadcast_sd(&a_packed[offsetA + 10 * a_pack_cols]);
        t[60] = _mm256_fmadd_pd(a10, b0, t[60]);
        t[61] = _mm256_fmadd_pd(a10, b1, t[61]);
        t[62] = _mm256_fmadd_pd(a10, b2, t[62]);
        t[63] = _mm256_fmadd_pd(a10, b3, t[63]);
        t[64] = _mm256_fmadd_pd(a10, b4, t[64]);
        t[65] = _mm256_fmadd_pd(a10, b5, t[65]);

        __m256d a11 = _mm256_broadcast_sd(&a_packed[offsetA + 11 * a_pack_cols]);
        t[66] = _mm256_fmadd_pd(a11, b0, t[66]);
        t[67] = _mm256_fmadd_pd(a11, b1, t[67]);
        t[68] = _mm256_fmadd_pd(a11, b2, t[68]);
        t[69] = _mm256_fmadd_pd(a11, b3, t[69]);
        t[70] = _mm256_fmadd_pd(a11, b4, t[70]);
        t[71] = _mm256_fmadd_pd(a11, b5, t[71]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 6 + j][k];
            }
        }
    }
}

void kernel_12x32_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                       const int b_pack_cols, const int a_idx, const int a_pack_idx,
                       const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(96);
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 96; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_packed[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_packed[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_packed[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_packed[offsetB + 12]);
        __m256d b4 = _mm256_load_pd(&b_packed[offsetB + 16]);
        __m256d b5 = _mm256_load_pd(&b_packed[offsetB + 20]);
        __m256d b6 = _mm256_load_pd(&b_packed[offsetB + 24]);
        __m256d b7 = _mm256_load_pd(&b_packed[offsetB + 28]);

        __m256d a0 = _mm256_broadcast_sd(&a_packed[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);
        t[5] = _mm256_fmadd_pd(a0, b5, t[5]);
        t[6] = _mm256_fmadd_pd(a0, b6, t[6]);
        t[7] = _mm256_fmadd_pd(a0, b7, t[7]);

        __m256d a1 = _mm256_broadcast_sd(&a_packed[offsetA + a_pack_cols]);
        t[8] = _mm256_fmadd_pd(a1, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a1, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a1, b3, t[11]);
        t[12] = _mm256_fmadd_pd(a1, b4, t[12]);
        t[13] = _mm256_fmadd_pd(a1, b5, t[13]);
        t[14] = _mm256_fmadd_pd(a1, b6, t[14]);
        t[15] = _mm256_fmadd_pd(a1, b7, t[15]);

        __m256d a2 = _mm256_broadcast_sd(&a_packed[offsetA + 2 * a_pack_cols]);
        t[16] = _mm256_fmadd_pd(a2, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a2, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a2, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a2, b3, t[19]);
        t[20] = _mm256_fmadd_pd(a2, b4, t[20]);
        t[21] = _mm256_fmadd_pd(a2, b5, t[21]);
        t[22] = _mm256_fmadd_pd(a2, b6, t[22]);
        t[23] = _mm256_fmadd_pd(a2, b7, t[23]);

        __m256d a3 = _mm256_broadcast_sd(&a_packed[offsetA + 3 * a_pack_cols]);
        t[24] = _mm256_fmadd_pd(a3, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a3, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a3, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a3, b3, t[27]);
        t[28] = _mm256_fmadd_pd(a3, b4, t[28]);
        t[29] = _mm256_fmadd_pd(a3, b5, t[29]);
        t[30] = _mm256_fmadd_pd(a3, b6, t[30]);
        t[31] = _mm256_fmadd_pd(a3, b7, t[31]);

        __m256d a4 = _mm256_broadcast_sd(&a_packed[offsetA + 4 * a_pack_cols]);
        t[32] = _mm256_fmadd_pd(a4, b0, t[32]);
        t[33] = _mm256_fmadd_pd(a4, b1, t[33]);
        t[34] = _mm256_fmadd_pd(a4, b2, t[34]);
        t[35] = _mm256_fmadd_pd(a4, b3, t[35]);
        t[36] = _mm256_fmadd_pd(a4, b4, t[36]);
        t[37] = _mm256_fmadd_pd(a4, b5, t[37]);
        t[38] = _mm256_fmadd_pd(a4, b6, t[38]);
        t[39] = _mm256_fmadd_pd(a4, b7, t[39]);

        __m256d a5 = _mm256_broadcast_sd(&a_packed[offsetA + 5 * a_pack_cols]);
        t[40] = _mm256_fmadd_pd(a5, b0, t[40]);
        t[41] = _mm256_fmadd_pd(a5, b1, t[41]);
        t[42] = _mm256_fmadd_pd(a5, b2, t[42]);
        t[43] = _mm256_fmadd_pd(a5, b3, t[43]);
        t[44] = _mm256_fmadd_pd(a5, b4, t[44]);
        t[45] = _mm256_fmadd_pd(a5, b5, t[45]);
        t[46] = _mm256_fmadd_pd(a5, b6, t[46]);
        t[47] = _mm256_fmadd_pd(a5, b7, t[47]);

        __m256d a6 = _mm256_broadcast_sd(&a_packed[offsetA + 6 * a_pack_cols]);
        t[48] = _mm256_fmadd_pd(a6, b0, t[48]);
        t[49] = _mm256_fmadd_pd(a6, b1, t[49]);
        t[50] = _mm256_fmadd_pd(a6, b2, t[50]);
        t[51] = _mm256_fmadd_pd(a6, b3, t[51]);
        t[52] = _mm256_fmadd_pd(a6, b4, t[52]);
        t[53] = _mm256_fmadd_pd(a6, b5, t[53]);
        t[54] = _mm256_fmadd_pd(a6, b6, t[54]);
        t[55] = _mm256_fmadd_pd(a6, b7, t[55]);

        __m256d a7 = _mm256_broadcast_sd(&a_packed[offsetA + 7 * a_pack_cols]);
        t[56] = _mm256_fmadd_pd(a7, b0, t[56]);
        t[57] = _mm256_fmadd_pd(a7, b1, t[57]);
        t[58] = _mm256_fmadd_pd(a7, b2, t[58]);
        t[59] = _mm256_fmadd_pd(a7, b3, t[59]);
        t[60] = _mm256_fmadd_pd(a7, b4, t[60]);
        t[61] = _mm256_fmadd_pd(a7, b5, t[61]);
        t[62] = _mm256_fmadd_pd(a7, b6, t[62]);
        t[63] = _mm256_fmadd_pd(a7, b7, t[63]);

        __m256d a8 = _mm256_broadcast_sd(&a_packed[offsetA + 8 * a_pack_cols]);
        t[64] = _mm256_fmadd_pd(a8, b0, t[64]);
        t[65] = _mm256_fmadd_pd(a8, b1, t[65]);
        t[66] = _mm256_fmadd_pd(a8, b2, t[66]);
        t[67] = _mm256_fmadd_pd(a8, b3, t[67]);
        t[68] = _mm256_fmadd_pd(a8, b4, t[68]);
        t[69] = _mm256_fmadd_pd(a8, b5, t[69]);
        t[70] = _mm256_fmadd_pd(a8, b6, t[70]);
        t[71] = _mm256_fmadd_pd(a8, b7, t[71]);

        __m256d a9 = _mm256_broadcast_sd(&a_packed[offsetA + 9 * a_pack_cols]);
        t[72] = _mm256_fmadd_pd(a9, b0, t[72]);
        t[73] = _mm256_fmadd_pd(a9, b1, t[73]);
        t[74] = _mm256_fmadd_pd(a9, b2, t[74]);
        t[75] = _mm256_fmadd_pd(a9, b3, t[75]);
        t[76] = _mm256_fmadd_pd(a9, b4, t[76]);
        t[77] = _mm256_fmadd_pd(a9, b5, t[77]);
        t[78] = _mm256_fmadd_pd(a9, b6, t[78]);
        t[79] = _mm256_fmadd_pd(a9, b7, t[79]);

        __m256d a10 = _mm256_broadcast_sd(&a_packed[offsetA + 10 * a_pack_cols]);
        t[80] = _mm256_fmadd_pd(a10, b0, t[80]);
        t[81] = _mm256_fmadd_pd(a10, b1, t[81]);
        t[82] = _mm256_fmadd_pd(a10, b2, t[82]);
        t[83] = _mm256_fmadd_pd(a10, b3, t[83]);
        t[84] = _mm256_fmadd_pd(a10, b4, t[84]);
        t[85] = _mm256_fmadd_pd(a10, b5, t[85]);
        t[86] = _mm256_fmadd_pd(a10, b6, t[86]);
        t[87] = _mm256_fmadd_pd(a10, b7, t[87]);

        __m256d a11 = _mm256_broadcast_sd(&a_packed[offsetA + 11 * a_pack_cols]);
        t[88] = _mm256_fmadd_pd(a11, b0, t[88]);
        t[89] = _mm256_fmadd_pd(a11, b1, t[89]);
        t[90] = _mm256_fmadd_pd(a11, b2, t[90]);
        t[91] = _mm256_fmadd_pd(a11, b3, t[91]);
        t[92] = _mm256_fmadd_pd(a11, b4, t[92]);
        t[93] = _mm256_fmadd_pd(a11, b5, t[93]);
        t[94] = _mm256_fmadd_pd(a11, b6, t[94]);
        t[95] = _mm256_fmadd_pd(a11, b7, t[95]);

        offsetA++;
        offsetB += b_pack_cols;
    }

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 8 + j][k];
            }
        }
    }
}
