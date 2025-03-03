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


void kernel_4x12_changed_order3(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
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



void kernel_omp_8x16_v1(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_omp_8x16_v2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_omp_8x16_v3(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_T_v4(double *a_aligned, double *b_aligned_transposed, double *c_aligned, const int d, const int a_rows,
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
void kernel2(double *aligned_a, double *aligned_b, double *c, const int bd, const int a_rows, const int b_cols,
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
void kernel_var(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
                const int b_cols,
                const int a_cols, int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) {
    // Create a vector of __m256d with size height * width
    aligned_vector<__m256d> t(h * wl);
    // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
    // t[i]: i-th element of t (a 4 element _m256d SIMD vector), t[i][j]: j-th double of i-th element of t

    // Initialize each element with zeros
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < h * wl; ++i) {
        t[i] = zero;
    }

    int offsetA = (d * a_rows + a_idx) * a_cols + l;
    int offsetB = d * a_cols * b_cols + b_idx + l * b_cols;
    for (int k = l; k < r; k++) {
        for (int j = 0; j < wl; j++) {
            //            cout << "k: " << k << " j: " << j << endl;
            __m256d b0 = _mm256_load_pd(&b_aligned[d * a_cols * b_cols + k * b_cols + b_idx + j * simd_length]);
            //            cout << "b0: " << b0[0] << " " << b0[1] << " " << b0[2] << " " << b0[3] << endl;
            for (int i = 0; i < h; i++) {
                //                cout << "i: " << i << endl;
                __m256d a0 = _mm256_broadcast_sd(&a_aligned[d * a_rows * a_cols + (a_idx + i) * a_cols + k]);
                //                cout << "a0: " << a0[0] << " " << a0[1] << " " << a0[2] << " " << a0[3] << endl;
                t[i * wl + j] = _mm256_fmadd_pd(a0, b0, t[i * wl + j]);
                //                cout << "t[" << i * wl + j << "]: " << t[i * wl + j][0] << " " << t[i * wl + j][1] << " " << t[i * wl + j][2] << " " << t[i * wl + j][3] << endl;
            }
        }
    }

    int offsetC = (d * a_rows + a_idx) * b_cols + b_idx;
    // Update c with the values in t
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < wl; ++j) {
            for (int k = 0; k < simd_length; ++k) {
                c_aligned[d * a_rows * b_cols + (a_idx + i) * b_cols + b_idx + j * simd_length + k] += t[i * wl + j][k];
            }
        }
    }

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


    //    // Update c with the values in t, c += t
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

void kernel_2x24(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x12_test1(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x12_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_4x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_6x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_6x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_6x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_6x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_6x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x4(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x8(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x12(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x16(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x16_test(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x16_test2(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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

void kernel_8x16_pack(const double *a_packed, const double *b_packed, double *c_aligned, const int c_cols,
                      const int b_pack_cols, const int a_idx, const int b_idx, const int a_pack_idx,
                      const int b_pack_idx, const int l, const int r) {
    aligned_vector<__m256d> t(32); // h * wl = 8 * 4 = 32
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < 32; ++i) {
        t[i] = zero;
    }

    int a_pack_cols = r - l;
    int offsetA = a_pack_idx * a_pack_cols;
    int offsetB = b_pack_idx;
    for (int k = l; k < r; k++) {
        //_mm_prefetch(&b_aligned[offsetB + b_pack_cols], _MM_HINT_T0);

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
                c_aligned[(a_idx + i) * c_cols + b_pack_idx + j * 4 + k] += t[i * 4 + j][k];
            }
        }
    }
}


void kernel_8x20(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows,
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
