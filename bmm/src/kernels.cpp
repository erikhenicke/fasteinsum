#include "aligned_allocator.h"
#include "bmm.h"
#include <vector>  // std::vector for using aligned_vector
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <functional>
#include <omp.h>

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

using namespace std;



void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)) {
//         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int, int, int, int, int)) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    // #pragma omp parallel for
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
//                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols), h, w, simd_length, wl);
	                        kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols));
                        }
                    }
                }
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[d * a_rows * b_cols + i * b_cols], &c_aligned[d * a_rows_padded * b_cols_padded + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}

void bmm_parallel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int, int, int)) {

    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
	                        kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols), h, w);
                        }
                    }
                }
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[d * a_rows * b_cols + i * b_cols], &c_aligned[d * a_rows_padded * b_cols_padded + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}

void bmm_parallel_more4(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i1 = 0; i1 < a_cols; i1 += b1) {
                #pragma omp parallel for collapse(3)
                for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                    for (int k = i2; k < i2 + b2; k += h) {
                        for (int j = i3; j < i3 + b3; j += w) {
                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, i1 + b1);
                        }
                    }
                }
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[d * a_rows * b_cols + i * b_cols], &c_aligned[d * a_rows_padded * b_cols_padded + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}

void bmm_parallel_more5(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int d = 0; d < bd; ++d) {
        for (int i1 = 0; i1 < a_cols; i1 += b1) {
            #pragma omp parallel for collapse(4)
            for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
                for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                    for (int k = i2; k < i2 + b2; k += h) {
                        for (int j = i3; j < i3 + b3; j += w) {
                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, i1 + b1);
                        }
                    }
                }
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[d * a_rows * b_cols + i * b_cols], &c_aligned[d * a_rows_padded * b_cols_padded + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}

// kernel with hardcoded h, w

void kernel_2x24(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
                 int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(12); // h * wl = 2 * 6 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);
        __m256d b4 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 16]);
        __m256d b5 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 20]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);
        t[5] = _mm256_fmadd_pd(a0, b5, t[5]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a1, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b2, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b3, t[9]);
        t[10] = _mm256_fmadd_pd(a1, b4, t[10]);
        t[11] = _mm256_fmadd_pd(a1, b5, t[11]);
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 6 + j][k];
            }
        }
    }
}

void kernel_4x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(4); // h * wl = 4 * 1 = 4
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 4; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[1] = _mm256_fmadd_pd(a1, b0, t[1]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a2, b0, t[2]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[3] = _mm256_fmadd_pd(a3, b0, t[3]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 4; ++k) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + k] += t[i][k];
        }
    }
}

void kernel_4x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(8); // h * wl = 4 * 2 = 8
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 8; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a1, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a1, b1, t[3]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a2, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a2, b1, t[5]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a3, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 2 + j][k];
            }
        }
    }
}

void kernel_4x12_test1(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(12); // h * wl = 4 * 3 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);
        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);

        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);

        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        offsetA++;
        offsetB += b_cols;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_4x12_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    // aligned_vector<__m256d> t(12);
    __m256d* t = static_cast<__m256d*>(_mm_malloc(12 * sizeof(__m256d), 64));
    // h * wl = 4 * 3 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;
    for (int k = l; k < r; k++) {
        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);

        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);

        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);

        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        offsetA++;
        offsetB += b_cols;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }

    _mm_free(t);  // Don't forget to free!
}

void kernel_4x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(12); // h * wl = 4 * 3 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;
    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);
        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);
        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        offsetA++;
        offsetB += b_cols;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_4x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(16); // h * wl = 4 * 4 = 16
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 16; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_4x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(20); // h * wl = 4 * 5 = 20
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 20; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);
        __m256d b4 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 16]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[5] = _mm256_fmadd_pd(a1, b0, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b1, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b2, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b3, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b4, t[9]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a2, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b1, t[11]);
        t[12] = _mm256_fmadd_pd(a2, b2, t[12]);
        t[13] = _mm256_fmadd_pd(a2, b3, t[13]);
        t[14] = _mm256_fmadd_pd(a2, b4, t[14]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[15] = _mm256_fmadd_pd(a3, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a3, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a3, b2, t[17]);
        t[18] = _mm256_fmadd_pd(a3, b3, t[18]);
        t[19] = _mm256_fmadd_pd(a3, b4, t[19]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 5 + j][k];
            }
        }
    }
}

void kernel_6x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(6); // h * wl = 6 * 1 = 6
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 6; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[1] = _mm256_fmadd_pd(a1, b0, t[1]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a2, b0, t[2]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[3] = _mm256_fmadd_pd(a3, b0, t[3]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a4, b0, t[4]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[5] = _mm256_fmadd_pd(a5, b0, t[5]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 4; ++k) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + k] += t[i][k];
        }
    }
}

void kernel_6x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(12); // h * wl = 6 * 2 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a1, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a1, b1, t[3]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a2, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a2, b1, t[5]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a3, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[8] = _mm256_fmadd_pd(a4, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a4, b1, t[9]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a5, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a5, b1, t[11]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 2 + j][k];
            }
        }
    }
}

void kernel_6x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(18); // h * wl = 6 * 3 = 18
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 18; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[12] = _mm256_fmadd_pd(a4, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a4, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a4, b2, t[14]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[15] = _mm256_fmadd_pd(a5, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a5, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a5, b2, t[17]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_6x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(24); // h * wl = 6 * 4 = 24
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 24; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}

void kernel_6x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(30); // h * wl = 6 * 5 = 30
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 30; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);
        __m256d b4 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 16]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[5] = _mm256_fmadd_pd(a1, b0, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b1, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b2, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b3, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b4, t[9]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a2, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b1, t[11]);
        t[12] = _mm256_fmadd_pd(a2, b2, t[12]);
        t[13] = _mm256_fmadd_pd(a2, b3, t[13]);
        t[14] = _mm256_fmadd_pd(a2, b4, t[14]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[15] = _mm256_fmadd_pd(a3, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a3, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a3, b2, t[17]);
        t[18] = _mm256_fmadd_pd(a3, b3, t[18]);
        t[19] = _mm256_fmadd_pd(a3, b4, t[19]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[20] = _mm256_fmadd_pd(a4, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a4, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a4, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a4, b3, t[23]);
        t[24] = _mm256_fmadd_pd(a4, b4, t[24]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[25] = _mm256_fmadd_pd(a5, b0, t[25]);
        t[26] = _mm256_fmadd_pd(a5, b1, t[26]);
        t[27] = _mm256_fmadd_pd(a5, b2, t[27]);
        t[28] = _mm256_fmadd_pd(a5, b3, t[28]);
        t[29] = _mm256_fmadd_pd(a5, b4, t[29]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 5 + j][k];
            }
        }
    }
}

void kernel_8x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(8); // h * wl = 8 * 1 = 8
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 8; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[1] = _mm256_fmadd_pd(a1, b0, t[1]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a2, b0, t[2]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[3] = _mm256_fmadd_pd(a3, b0, t[3]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a4, b0, t[4]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[5] = _mm256_fmadd_pd(a5, b0, t[5]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 6) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a6, b0, t[6]);

        __m256d a7 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 7) * a_cols + k]);
        t[7] = _mm256_fmadd_pd(a7, b0, t[7]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 4; ++k) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + k] += t[i][k];
        }
    }
}

void kernel_8x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(16); // h * wl = 8 * 2 = 16
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 16; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a1, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a1, b1, t[3]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a2, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a2, b1, t[5]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a3, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[8] = _mm256_fmadd_pd(a4, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a4, b1, t[9]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a5, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a5, b1, t[11]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 6) * a_cols + k]);
        t[12] = _mm256_fmadd_pd(a6, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a6, b1, t[13]);

        __m256d a7 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 7) * a_cols + k]);
        t[14] = _mm256_fmadd_pd(a7, b0, t[14]);
        t[15] = _mm256_fmadd_pd(a7, b1, t[15]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 2 + j][k];
            }
        }
    }
}

void kernel_8x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(24); // h * wl = 8 * 3 = 24
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 24; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[3] = _mm256_fmadd_pd(a1, b0, t[3]);
        t[4] = _mm256_fmadd_pd(a1, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b2, t[5]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a2, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a2, b1, t[7]);
        t[8] = _mm256_fmadd_pd(a2, b2, t[8]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[9] = _mm256_fmadd_pd(a3, b0, t[9]);
        t[10] = _mm256_fmadd_pd(a3, b1, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[12] = _mm256_fmadd_pd(a4, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a4, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a4, b2, t[14]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[15] = _mm256_fmadd_pd(a5, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a5, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a5, b2, t[17]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 6) * a_cols + k]);
        t[18] = _mm256_fmadd_pd(a6, b0, t[18]);
        t[19] = _mm256_fmadd_pd(a6, b1, t[19]);
        t[20] = _mm256_fmadd_pd(a6, b2, t[20]);

        __m256d a7 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 7) * a_cols + k]);
        t[21] = _mm256_fmadd_pd(a7, b0, t[21]);
        t[22] = _mm256_fmadd_pd(a7, b1, t[22]);
        t[23] = _mm256_fmadd_pd(a7, b2, t[23]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 3 + j][k];
            }
        }
    }
}

void kernel_8x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r, int h, int w) {
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

void kernel_8x16_test(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d* t = static_cast<__m256d*>(_mm_malloc(32 * sizeof(__m256d), 64));

    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

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

    _mm_free(t);
}

void kernel_8x16_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d* t = static_cast<__m256d*>(_mm_malloc(32 * sizeof(__m256d), 64));

    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {

        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[offsetB + 12]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);
        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);

        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);

        t[4] = _mm256_fmadd_pd(a1, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b2, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b3, t[7]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);

        t[8] = _mm256_fmadd_pd(a2, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a2, b1, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b3, t[11]);

        t[12] = _mm256_fmadd_pd(a3, b0, t[12]);
        t[13] = _mm256_fmadd_pd(a3, b1, t[13]);
        t[14] = _mm256_fmadd_pd(a3, b2, t[14]);
        t[15] = _mm256_fmadd_pd(a3, b3, t[15]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[offsetA + 4 * a_cols]);
        __m256d a5 = _mm256_broadcast_sd(&a_aligned[offsetA + 5 * a_cols]);

        t[16] = _mm256_fmadd_pd(a4, b0, t[16]);
        t[17] = _mm256_fmadd_pd(a4, b1, t[17]);
        t[18] = _mm256_fmadd_pd(a4, b2, t[18]);
        t[19] = _mm256_fmadd_pd(a4, b3, t[19]);

        t[20] = _mm256_fmadd_pd(a5, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a5, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a5, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a5, b3, t[23]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[offsetA + 6 * a_cols]);
        __m256d a7 = _mm256_broadcast_sd(&a_aligned[offsetA + 7 * a_cols]);

        t[24] = _mm256_fmadd_pd(a6, b0, t[24]);
        t[25] = _mm256_fmadd_pd(a6, b1, t[25]);
        t[26] = _mm256_fmadd_pd(a6, b2, t[26]);
        t[27] = _mm256_fmadd_pd(a6, b3, t[27]);

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

    _mm_free(t);
}


void kernel_8x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
                 int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(40); // h * wl = 8 * 5 = 40
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 40; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 4]);
        __m256d b2 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 8]);
        __m256d b3 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 12]);
        __m256d b4 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + 16]);

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);
        t[2] = _mm256_fmadd_pd(a0, b2, t[2]);
        t[3] = _mm256_fmadd_pd(a0, b3, t[3]);
        t[4] = _mm256_fmadd_pd(a0, b4, t[4]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 1) * a_cols + k]);
        t[5] = _mm256_fmadd_pd(a1, b0, t[5]);
        t[6] = _mm256_fmadd_pd(a1, b1, t[6]);
        t[7] = _mm256_fmadd_pd(a1, b2, t[7]);
        t[8] = _mm256_fmadd_pd(a1, b3, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b4, t[9]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 2) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a2, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a2, b1, t[11]);
        t[12] = _mm256_fmadd_pd(a2, b2, t[12]);
        t[13] = _mm256_fmadd_pd(a2, b3, t[13]);
        t[14] = _mm256_fmadd_pd(a2, b4, t[14]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 3) * a_cols + k]);
        t[15] = _mm256_fmadd_pd(a3, b0, t[15]);
        t[16] = _mm256_fmadd_pd(a3, b1, t[16]);
        t[17] = _mm256_fmadd_pd(a3, b2, t[17]);
        t[18] = _mm256_fmadd_pd(a3, b3, t[18]);
        t[19] = _mm256_fmadd_pd(a3, b4, t[19]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 4) * a_cols + k]);
        t[20] = _mm256_fmadd_pd(a4, b0, t[20]);
        t[21] = _mm256_fmadd_pd(a4, b1, t[21]);
        t[22] = _mm256_fmadd_pd(a4, b2, t[22]);
        t[23] = _mm256_fmadd_pd(a4, b3, t[23]);
        t[24] = _mm256_fmadd_pd(a4, b4, t[24]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 5) * a_cols + k]);
        t[25] = _mm256_fmadd_pd(a5, b0, t[25]);
        t[26] = _mm256_fmadd_pd(a5, b1, t[26]);
        t[27] = _mm256_fmadd_pd(a5, b2, t[27]);
        t[28] = _mm256_fmadd_pd(a5, b3, t[28]);
        t[29] = _mm256_fmadd_pd(a5, b4, t[29]);

        __m256d a6 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 6) * a_cols + k]);
        t[30] = _mm256_fmadd_pd(a6, b0, t[30]);
        t[31] = _mm256_fmadd_pd(a6, b1, t[31]);
        t[32] = _mm256_fmadd_pd(a6, b2, t[32]);
        t[33] = _mm256_fmadd_pd(a6, b3, t[33]);
        t[34] = _mm256_fmadd_pd(a6, b4, t[34]);

        __m256d a7 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + 7) * a_cols + k]);
        t[35] = _mm256_fmadd_pd(a7, b0, t[35]);
        t[36] = _mm256_fmadd_pd(a7, b1, t[36]);
        t[37] = _mm256_fmadd_pd(a7, b2, t[37]);
        t[38] = _mm256_fmadd_pd(a7, b3, t[38]);
        t[39] = _mm256_fmadd_pd(a7, b4, t[39]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j * 4 + k] += t[i * 5 + j][k];
            }
        }
    }
}


//int main() {
//    // Define matrix dimensions
//    const int bd = 4;
//    const int a_rows = 256;
//    const int b_cols = 256;
//    const int a_cols = 256;
//    const int h = 4;
//    const int w = 20;
//    const int simd_length = 4;
//    const int wl = w / simd_length;
//    const int b1 = 32;
//    const int b2_ = 64;
//    const int b3_ = 128;
//
//    // Allocate memory for matrices A, B, and C
//    aligned_vector<double> a(bd * a_rows * a_cols);
//    aligned_vector<double> b(bd * a_cols * b_cols);
//    aligned_vector<double> c(bd * a_rows * b_cols, 0.0);
//
//    // Generate random matrices A and B
//    generate_random_matrix(a, a_rows, a_cols);
//    generate_random_matrix(b, a_cols, b_cols);
//
//    cout << "Starting bmm" << endl;
//
//    // Call bmm with kernel_6x8
//    bmm(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, h, w, simd_length, wl, b1, b2_, b3_, kernel_4x20);
//
//    cout << "Finished bmm" << endl;
//
//    // Compare this with the reference implementation
//    aligned_vector<double> c_ref(bd * a_rows * b_cols, 0.0);
//    bmm_naive(a.data(), b.data(), c_ref.data(), bd, a_rows, b_cols, a_cols);
//
//
//    // Check if the results are correct
//    bool equal = true;
//    for (int i = 0; i < bd * a_rows * b_cols; ++i) {
//        if (abs(c[i] - c_ref[i]) > 1e-6) {
//            equal = false;
//            break;
//        }
//    }
//
//    cout << "Result of bmm is " << (equal ? "correct" : "incorrect") << endl;
//
//    return 0;
//}