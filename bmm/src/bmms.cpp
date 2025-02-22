//
// Created by sonja on 06.02.25.
//

#include "bmms.h"
#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <random>
//#include <chrono>
//#include <fstream>
#include <iostream>

// 1. bmm with kernel hxw = {6x8, 4x12, 8x16, 8x20}
// 2. bmm with pipelining kernel
// 3. bmm with blocking only
// 4. bmm with OMP SIMD



// 1. pipelining kernel "kernel2"
void pack2(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded) {
    // transposes B and pads A and B

      // Allocate aligned memory for matrices A, B, and C with padded dimensions
    a_aligned.resize(bd * a_rows_padded * a_cols);
    b_aligned.resize(bd * a_cols * b_cols_padded);
    c_aligned.resize(bd * a_rows_padded * b_cols_padded);

    // Copy data from original matrix A to aligned memory and set padded elements to 0
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_rows_padded; ++i) {
            if (i < a_rows) {
                std::memcpy(&a_aligned[(d * a_rows_padded + i) * a_cols], &a[(d * a_rows + i) * a_cols], a_cols * sizeof(double));
            } else {
                std::fill(&a_aligned[(d * a_rows_padded + i) * a_cols], &a_aligned[(d * a_rows_padded + (i + 1)) * a_cols], 0.0);
            }
        }
    }

    // Copy data from original matrix B to aligned memory, while transposing, and set padded elements to 0
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_cols; ++i) {
            if (i < a_cols) {
                for (int j = 0; j < b_cols; ++j) {
                    b_aligned[d * a_cols * b_cols_padded + i * b_cols + j] = b[d * a_cols * b_cols + j * b_cols + i];
                }
                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols + b_cols], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols], 0.0);
            } else {
                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols], 0.0);
            }
        }
    }


    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}

void kernel2(double *a_aligned, double *b_aligned, double *c, const int d, const int a_rows, const int b_cols,
             const int a_cols, int a_idx, int b_idx, int l, int r, int height, int width) {
    // Assumption: A and B already aligned and B transposed
    // C gets updated - Initialization happens before.
    // Update C (add to c) for cols l to r from a and rows l to r from b (In the end from l=0 to r=a_cols must be called)

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = a_idx; i < a_idx + height; ++i) {
        for (int j = b_idx; j < b_idx + width; ++j) {
            __m256d c_vec1 = _mm256_setzero_pd();
            __m256d c_vec2 = _mm256_setzero_pd();
            __m256d c_vec3 = _mm256_setzero_pd();
            __m256d c_vec4 = _mm256_setzero_pd();
            __m256d c_vec5 = _mm256_setzero_pd();
            __m256d c_vec6 = _mm256_setzero_pd();
            __m256d c_vec7 = _mm256_setzero_pd();
            __m256d c_vec8 = _mm256_setzero_pd();

            // Offset for a and b
            int a_offset = (d * a_rows + i) * a_cols;
    		int b_offset = (d * b_cols + j) * a_cols; // because B is transposed


            for (int k = l; k < r; k += 32) { // 32 = 8 * 4 = simd_length * #temp_regs
//                _mm_prefetch((const char*)&a_aligned[a_offset + k + 32], _MM_HINT_T0);
//                _mm_prefetch((const char*)&b_aligned[b_offset + k + 32], _MM_HINT_T0);

                __m256d a_vec1 = _mm256_load_pd(&a_aligned[a_offset + k]);
                __m256d b_vec1 = _mm256_load_pd(&b_aligned[b_offset + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&a_aligned[a_offset + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&b_aligned[b_offset + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&a_aligned[a_offset + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&b_aligned[b_offset + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&a_aligned[a_offset + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&b_aligned[b_offset + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&a_aligned[a_offset + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&b_aligned[b_offset + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&a_aligned[a_offset + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&b_aligned[b_offset + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&a_aligned[a_offset + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&b_aligned[b_offset + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&a_aligned[a_offset + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&b_aligned[b_offset + k + 28]);
                c_vec8 = _mm256_fmadd_pd(a_vec8, b_vec8, c_vec8);
            }
            // Add all entries of c_vec1 to c_vec8 using horizontal add
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
            c[i * b_cols + j] += c_low[0] + c_low[1];
        }
    }
}

void bmm2(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int b1, int b2_, int b3_) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    //    int b1 = b1_ - (b1_ % simd_length);
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_transposed;
    aligned_vector<double> c_aligned;
    pack2(a, b, a_aligned, b_aligned_transposed, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    #pragma omp parallel for // collapse(4) // TODO: update c right always???
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
                            kernel2(a_aligned.data(), b_aligned_transposed.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols), h, w);
                        }
                    }
                }
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < bd; ++d) {
        int offsetC = d * a_rows * b_cols;
        int offsetCAligned = d * a_rows_padded * b_cols_padded;
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[offsetC + i * b_cols], &c_aligned[offsetCAligned + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}