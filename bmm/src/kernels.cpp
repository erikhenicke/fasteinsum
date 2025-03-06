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


void kernel_4x12_changed_order3(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                                const int a_rows, const int b_cols, const int a_cols,
                                int a_idx, int b_idx, int l, int r) {
    aligned_vector<__m256d> t(12); // h * wl = 4 * 3 = 12
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 12; ++i) {
        t[i] = zero;
    }

    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;
    for (int k = l; k < r; k++) {
        __m256d a0 = _mm256_broadcast_sd(&a_aligned[offsetA]);
        __m256d a1 = _mm256_broadcast_sd(&a_aligned[offsetA + 1 * a_cols]);
        __m256d a2 = _mm256_broadcast_sd(&a_aligned[offsetA + 2 * a_cols]);
        __m256d a3 = _mm256_broadcast_sd(&a_aligned[offsetA + 3 * a_cols]);

        __m256d b0 = _mm256_load_pd(&b_aligned[offsetB]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a1, b0, t[1]);
        t[2] = _mm256_fmadd_pd(a2, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a3, b0, t[3]);

        __m256d b1 = _mm256_load_pd(&b_aligned[offsetB + 4]);
        t[4] = _mm256_fmadd_pd(a0, b1, t[4]);
        t[5] = _mm256_fmadd_pd(a1, b1, t[5]);
        t[6] = _mm256_fmadd_pd(a2, b1, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);

        __m256d b2 = _mm256_load_pd(&b_aligned[offsetB + 8]);
        t[8] = _mm256_fmadd_pd(a0, b2, t[8]);
        t[9] = _mm256_fmadd_pd(a1, b2, t[9]);
        t[10] = _mm256_fmadd_pd(a2, b2, t[10]);
        t[11] = _mm256_fmadd_pd(a3, b2, t[11]);

        offsetA++;
        offsetB += b_cols;
    }
    // indexing changed
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_aligned[(d * a_rows + a_idx + j) * b_cols + b_idx + i * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}


void kernel_omp_8x16_v1(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                        const int a_rows,
                        const int b_cols, const int a_cols,
                        int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // Temporary storage for the results
    double t[8][16] = {0};

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            t[i][j] = 0.0;
        }
    }

    for (int k = l; k < r; k++) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 16; ++j) {
                t[i][j] += a_aligned[offsetA + i * a_cols] * b_aligned[offsetB + j];
            }
        }
        offsetA++;
        offsetB += b_cols;
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j] += t[i][j];
        }
    }
}

void kernel_omp_8x16_v2(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                        const int a_rows,
                        const int b_cols, const int a_cols,
                        int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

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
                t[i][j] += a_aligned[offsetA + i * a_cols] * b_aligned[offsetB + j];
            }
        }
        offsetA++;
        offsetB += b_cols;
    }

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j] += t[i][j];
        }
    }
}

void kernel_omp_8x16_v3(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                        const int a_rows,
                        const int b_cols, const int a_cols,
                        int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // Temporary storage for the results
    double t[8][16] = {0};

    //    #pragma omp simd
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            t[i][j] = 0.0;
        }
    }

    for (int k = l; k < r; k++) {
        //        #pragma omp simd
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 16; ++j) {
                t[i][j] += a_aligned[offsetA + i * a_cols] * b_aligned[offsetB + j];
            }
        }
        offsetA++;
        offsetB += b_cols;
    }

    //    #pragma omp simd
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j] += t[i][j];
        }
    }
}

void kernel_T_v4(const double *a_aligned, const double *b_aligned_transposed, double *c_aligned, const int d,
                 const int a_rows,
                 const int b_cols, const int a_cols,
                 int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * b_cols * a_cols + b_idx * a_cols + l;

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
                t[i][j] += a_aligned[offsetA + i * a_cols] * b_aligned_transposed[offsetB + j * a_cols];
            }
        }
        offsetA++;
        offsetB++;
    }

    for (int i = 0; i < 8; ++i) {
#pragma omp simd
        for (int j = 0; j < 16; ++j) {
            c_aligned[(d * a_rows + a_idx + i) * b_cols + b_idx + j] += t[i][j];
        }
    }
}

// Simple kernel (with AVX2 but not as efficient as other kernels)
void simple_kernel(const double *aligned_a, const double *aligned_b, double *c, const int bd, const int a_rows,
                   const int b_cols,
                   const int a_cols,
                   int a_idx, int b_idx, int l, int r, int height, int width) {
    const int a_offset = bd * a_rows * a_cols;
    const int b_offset = bd * a_cols * b_cols;
    const int c_offset = bd * a_rows * b_cols;

    for (int i = a_idx; i < a_idx + height; ++i) {
        const int a_i_offset = a_offset + i * a_cols;
        const int c_i_offset = c_offset + i * b_cols;
        for (int j = b_idx; j < b_idx + width; ++j) {
            const int b_j_offset = b_offset + j * a_cols;
            __m256d c_vec1 = _mm256_setzero_pd();
            __m256d c_vec2 = _mm256_setzero_pd();
            __m256d c_vec3 = _mm256_setzero_pd();
            __m256d c_vec4 = _mm256_setzero_pd();
            __m256d c_vec5 = _mm256_setzero_pd();
            __m256d c_vec6 = _mm256_setzero_pd();
            __m256d c_vec7 = _mm256_setzero_pd();
            __m256d c_vec8 = _mm256_setzero_pd();
            for (int k = l; k < r; k += 32) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[a_i_offset + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[b_j_offset + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[a_i_offset + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[b_j_offset + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[a_i_offset + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[b_j_offset + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[a_i_offset + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[b_j_offset + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[a_i_offset + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[b_j_offset + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[a_i_offset + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[b_j_offset + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[a_i_offset + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[b_j_offset + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[a_i_offset + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[b_j_offset + k + 28]);
                c_vec8 = _mm256_fmadd_pd(a_vec8, b_vec8, c_vec8);
            }
            c_vec1 = _mm256_add_pd(c_vec1, c_vec2);
            c_vec3 = _mm256_add_pd(c_vec3, c_vec4);
            c_vec5 = _mm256_add_pd(c_vec5, c_vec6);
            c_vec7 = _mm256_add_pd(c_vec7, c_vec8);
            c_vec1 = _mm256_add_pd(c_vec1, c_vec3);
            c_vec5 = _mm256_add_pd(c_vec5, c_vec7);
            c_vec1 = _mm256_add_pd(c_vec1, c_vec5);
            __m128d c_low = _mm256_castpd256_pd128(c_vec1);
            __m128d c_high = _mm256_extractf128_pd(c_vec1, 1);
            c_low = _mm_add_pd(c_low, c_high);
            c[c_i_offset + j] += c_low[0] + c_low[1];
        }
    }
}

// kernel with h, w as parameters (no loop unrolling)
void kernel_var(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols,
                const int a_cols, int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) {
    // Create a vector of __m256d with size height * (width / simd_length)
    aligned_vector<__m256d> t(h * wl);
    // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
    // t[i]: i-th element of t (a 4 element _m256d SIMD vector), t[i][j]: j-th double of i-th element of t

    // Initialize each element with zeros
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < h * wl; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        for (int j = 0; j < wl; j++) {
            __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + j * simd_length]);
            for (int i = 0; i < h; i++) {
                __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + i) * a_cols + k]);
                t[i * wl + j] = _mm256_fmadd_pd(a0, b0, t[i * wl + j]);
            }
        }
    }

    // Update c with the values in t
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < wl; ++j) {
            for (int k = 0; k < simd_length; ++k) {
                c_aligned[d * a_rows * b_cols + (a_idx + i) * b_cols + b_idx + j * simd_length + k] += t[i * wl + j][k];
            }
        }
    }

    // NOTE: We tried different techniques to store the results in c.
    // int offsetC = (d * a_rows + a_idx) * b_cols + b_idx;
    // // Update c with the values in t
    // for (int i = 0; i < h; ++i) {
    //     for (int j = 0; j < wl; ++j) {
    //         for (int k = 0; k < simd_length; ++k) {
    //             c_aligned[offsetC + k] += t[i * wl + j][k];
    //         }
    //         offsetC += simd_length;
    //     }
    //     offsetC -= wl * simd_length;
    //     offsetC += b_cols;
    // }


    //  // Update c with the values in t, c += t
    //	for (int i = 0; i < h; ++i) {
    //    	for (int j = 0; j < wl; ++j) {
    //    	    __m256d c_val = _mm256_load_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length]);
    //    	    __m256d t_val = t[i * wl + j];
    //    	    __m256d result = _mm256_add_pd(c_val, t_val);
    //    	    _mm256_store_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length], result);
    //    	}
    //	}
}

// kernel with hardcoded h, w
void kernel_2x24(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_4x4(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_4x8(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_4x12_test1(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                       const int a_rows,
                       const int b_cols, const int a_cols,
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

void kernel_4x12_test2(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                       const int a_rows,
                       const int b_cols, const int a_cols,
                       int a_idx, int b_idx, int l, int r) {
    // aligned_vector<__m256d> t(12);
    __m256d *t = static_cast<__m256d *>(_mm_malloc(12 * sizeof(__m256d), 64));
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

    _mm_free(t); // Don't forget to free!
}

void kernel_4x12(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_4x16(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_4x20(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_6x4(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_6x8(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_6x12(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_6x16(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_6x20(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_8x4(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_8x8(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols, const int a_cols,
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

void kernel_8x12(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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

void kernel_8x16_test(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                      const int a_rows,
                      const int b_cols, const int a_cols,
                      int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d *t = static_cast<__m256d *>(_mm_malloc(32 * sizeof(__m256d), 64));

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

void kernel_8x16_test2(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d,
                       const int a_rows,
                       const int b_cols, const int a_cols,
                       int a_idx, int b_idx, int l, int r) {
    int offsetA = d * a_rows * a_cols + a_idx * a_cols + l;
    int offsetB = d * a_cols * b_cols + l * b_cols + b_idx;

    // aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d *t = static_cast<__m256d *>(_mm_malloc(32 * sizeof(__m256d), 64));

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

void kernel_8x20(const double *a_aligned, const double *b_aligned, double *c_aligned, const int d, const int a_rows,
                 const int b_cols, const int a_cols,
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
