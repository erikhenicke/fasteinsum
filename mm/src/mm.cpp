//
// Created by sonja on 10.01.25.
//

#include "mm.h"
#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;


// Naive implementation using three nested loops
void mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];
}

// Transpose matrix b so that is is in column-major order before multiplication
// Then column of b needed in each iteration is contiguous in memory -> better cache utilization
void mm_transposed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
  // Transpose matrix b
  double *b_transposed = new double[a_cols*b_cols];
    for (int i = 0; i < a_cols; ++i)
        for (int j = 0; j < b_cols; ++j)
            b_transposed[i*b_cols + j] = b[j*a_cols + i];
    // Naive MM (Note different indexing of b!)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b_transposed[j*a_cols + k];
    // Clean up
    delete[] b_transposed;
}

// Vectorization with SIMD instructions (automatically by compiler with __restrict__ keyword and manually)
void mm_auto_vectorized(const double *a, const double *b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols) {
    // Transpose matrix b
    double *b_transposed = new double[a_cols*b_cols];
    for (int i = 0; i < a_cols; ++i)
        for (int j = 0; j < b_cols; ++j)
            b_transposed[i*b_cols + j] = b[j*a_cols + i];
    // Naive MM (Note different indexing of b!)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b_transposed[j*a_cols + k];
    // Clean up
    delete[] b_transposed;
}

// align arrays to 64 bit cache line size
//void align_and_transpose_matrices(const double *a, const double *b, double **aligned_a, double **aligned_b, int a_rows, int a_cols, int b_cols) {
//    // Allocate aligned memory for matrices A and B
//    posix_memalign(reinterpret_cast<void**>(aligned_a), 64, a_rows * a_cols * sizeof(double));
//    posix_memalign(reinterpret_cast<void**>(aligned_b), 64, a_cols * b_cols * sizeof(double));
//
//    // Copy data from original matrix A to aligned memory
//    std::memcpy(*aligned_a, a, a_rows * a_cols * sizeof(double));
//
//    // Transpose and copy data from original matrix B to aligned memory
//    for (int i = 0; i < a_cols; ++i) {
//        for (int j = 0; j < b_cols; ++j) {
//            (*aligned_b)[i * b_cols + j] = b[j * a_cols + i];
//        }
//    }
//}

void mm_omp_vectorized(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            (aligned_b)[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    #pragma omp simd aligned(a, b, c: 64)
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += aligned_a[i*a_cols + k] * aligned_b[j*a_cols + k];
    // Clean up
    free(aligned_a);
    free(aligned_b);
}

// Vectorized implementation using AVX intrinsics
void mm_vectorized_64(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double)); //64?? 32- AVX instructions/64 CPU cache line
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double)); //64??

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

// Perform matrix multiplication using AVX intrinsics
for (int i = 0; i < a_rows; ++i) {
    // Loop over each row of matrix a
    for (int j = 0; j < b_cols; ++j) {
        // Loop over each column of matrix b
        __m256d c_vec = _mm256_setzero_pd(); // Initialize a vector of zeros
        // Initialize a 256-bit vector register with zeros to accumulate the results
        for (int k = 0; k < a_cols; k += 4) {
            // Loop over each element in the row of a and column of b in steps of 4
            __m256d a_vec = _mm256_load_pd(&aligned_a[i * a_cols + k]);
            // Load 4 double-precision elements from the current row of matrix a into a 256-bit vector register
            __m256d b_vec = _mm256_load_pd(&aligned_b[j * a_cols + k]);
            // Load 4 double-precision elements from the current column of matrix b (transposed) into a 256-bit vector register
            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            // Perform fused multiply-add: multiply elements of a_vec and b_vec, then add the result to c_vec
        }
        // Sum the elements of c_vec and store in c[i * b_cols + j]
        c[i * b_cols + j] += c_vec[0] + c_vec[1] + c_vec[2] + c_vec[3];
        // Sum the four elements in the 256-bit vector register and add the result to the corresponding element in matrix c
    }
}
    // Clean up
    free(aligned_a);
    free(aligned_b);
}

// Use ILP: Idea: use 0.5 Throughput of fmadd. pipeline fmadd (https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_pd&techs=AVX_ALL&ig_expand=3083)
void mm_vectorized_pipe_2(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            __m256d c_vec1 = _mm256_setzero_pd();
            __m256d c_vec2 = _mm256_setzero_pd();
            for (int k = 0; k < a_cols; k += 8) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);
            }
            c[i * b_cols + j] += c_vec1[0] + c_vec1[1] + c_vec1[2] + c_vec1[3] + c_vec2[0] + c_vec2[1] + c_vec2[2] + c_vec2[3];
        }
    }

    // Clean up
    free(aligned_a);
    free(aligned_b);
}

#include <immintrin.h>
#include <cstdlib>
#include <cstring>

void mm_vectorized_pipe_8(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            __m256d c_vec1 = _mm256_setzero_pd();
            __m256d c_vec2 = _mm256_setzero_pd();
            __m256d c_vec3 = _mm256_setzero_pd();
            __m256d c_vec4 = _mm256_setzero_pd();
            __m256d c_vec5 = _mm256_setzero_pd();
            __m256d c_vec6 = _mm256_setzero_pd();
            __m256d c_vec7 = _mm256_setzero_pd();
            __m256d c_vec8 = _mm256_setzero_pd();
            for (int k = 0; k < a_cols; k += 32) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[i * a_cols + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[j * a_cols + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[i * a_cols + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[j * a_cols + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[i * a_cols + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[j * a_cols + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[i * a_cols + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[j * a_cols + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[i * a_cols + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[j * a_cols + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[i * a_cols + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[j * a_cols + k + 28]);
                c_vec8 = _mm256_fmadd_pd(a_vec8, b_vec8, c_vec8);
            }
            c[i * b_cols + j] += c_vec1[0] + c_vec1[1] + c_vec1[2] + c_vec1[3] +
                                 c_vec2[0] + c_vec2[1] + c_vec2[2] + c_vec2[3] +
                                 c_vec3[0] + c_vec3[1] + c_vec3[2] + c_vec3[3] +
                                 c_vec4[0] + c_vec4[1] + c_vec4[2] + c_vec4[3] +
                                 c_vec5[0] + c_vec5[1] + c_vec5[2] + c_vec5[3] +
                                 c_vec6[0] + c_vec6[1] + c_vec6[2] + c_vec6[3] +
                                 c_vec7[0] + c_vec7[1] + c_vec7[2] + c_vec7[3] +
                                 c_vec8[0] + c_vec8[1] + c_vec8[2] + c_vec8[3];
        }
    }

    // Clean up
    free(aligned_a);
    free(aligned_b);
}

// Tried not transposing. Takes way longer, BAD IDEA

//void mm_vectorized_pipe_nT(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
//    // Allocate aligned memory for matrices A and B
//    double *aligned_a, *aligned_b;
//    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
//    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));
//
//    // Copy data from original matrix A to aligned memory
//    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));
//    std::memcpy(aligned_b, b, a_cols * b_cols * sizeof(double));
//
//    // Initialize result matrix c to zero
//    for (int i = 0; i < a_rows * b_cols; ++i)
//        c[i] = 0.0;
//
//    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
//    for (int i = 0; i < a_rows; ++i) {
//        for (int j = 0; j < b_cols; ++j) {
//            __m256d c_vec1 = _mm256_setzero_pd();
//            __m256d c_vec2 = _mm256_setzero_pd();
//            for (int k = 0; k < a_cols; k += 8) {
//                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
//                __m256d b_vec1 = _mm256_load_pd(&aligned_b[k * b_cols + j]);
//                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);
//
//                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
//                __m256d b_vec2 = _mm256_load_pd(&aligned_b[(k + 4) * b_cols + j]);
//                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);
//            }
//            c[i * b_cols + j] += c_vec1[0] + c_vec1[1] + c_vec1[2] + c_vec1[3] + c_vec2[0] + c_vec2[1] + c_vec2[2] + c_vec2[3];
//        }
//    }
//
//    // Clean up
//    free(aligned_a);
//    free(aligned_b);
//}


void kernel(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int height, int width, int l, int r) {
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
            for (int k = l; k < r; k += 32) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[i * a_cols + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[j * a_cols + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[i * a_cols + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[j * a_cols + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[i * a_cols + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[j * a_cols + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[i * a_cols + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[j * a_cols + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[i * a_cols + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[j * a_cols + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[i * a_cols + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[j * a_cols + k + 28]);
                c_vec8 = _mm256_fmadd_pd(a_vec8, b_vec8, c_vec8);
            }
            c[i * b_cols + j] += c_vec1[0] + c_vec1[1] + c_vec1[2] + c_vec1[3] +
                                 c_vec2[0] + c_vec2[1] + c_vec2[2] + c_vec2[3] +
                                 c_vec3[0] + c_vec3[1] + c_vec3[2] + c_vec3[3] +
                                 c_vec4[0] + c_vec4[1] + c_vec4[2] + c_vec4[3] +
                                 c_vec5[0] + c_vec5[1] + c_vec5[2] + c_vec5[3] +
                                 c_vec6[0] + c_vec6[1] + c_vec6[2] + c_vec6[3] +
                                 c_vec7[0] + c_vec7[1] + c_vec7[2] + c_vec7[3] +
                                 c_vec8[0] + c_vec8[1] + c_vec8[2] + c_vec8[3];
        }
    }
}

// Tried: Prefetching data. actually slows down

void kernel2(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int height, int width, int l, int r) {
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
            for (int k = l; k < r; k += 32) {
//                _mm_prefetch((const char*)&aligned_a[i * a_cols + k + 32], _MM_HINT_T0);
//                _mm_prefetch((const char*)&aligned_b[j * a_cols + k + 32], _MM_HINT_T0);

                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[i * a_cols + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[j * a_cols + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[i * a_cols + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[j * a_cols + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[i * a_cols + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[j * a_cols + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[i * a_cols + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[j * a_cols + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[i * a_cols + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[j * a_cols + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[i * a_cols + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[j * a_cols + k + 28]);
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
            c[i * b_cols + j] += c_low[0] + c_low[1];
        }
    }
}

// with batch dimension
void kernel3(double *aligned_a, double *aligned_b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int height, int width, int l, int r) {
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
            for (int k = l; k < r; k += 32) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[bd * a_rows * a_cols + i * a_cols + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[bd * a_cols * b_cols + j * a_cols + k + 28]);
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
            c[bd * a_rows * b_cols + i * b_cols + j] += c_low[0] + c_low[1];
        }
    }
}

// with offsets
void kernel4(double *aligned_a, double *aligned_b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int height, int width, int l, int r) {
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

void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel // TODO: try different kernel sizes
    int height = 4;
    int width = 4;
    // TODO: Pad Matrices so that dimensions are multiples of 4 and shared dim is multiple of 32

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows; i += height) { // TODO: use a_rows_padded here
        for (int j = 0; j < b_cols; j += width) { // TODO: use b_cols_padded here
//            std::cout << "i: " << i << " j: " << j << std::endl;
            kernel(aligned_a, aligned_b, c, a_rows, b_cols, a_cols, i, j, height, width, 0, a_cols);
            // for now l=0 and r=a_cols, later in block matrix multiplication, l and r will be updated
        }
    }

    // Clean up
    free(aligned_a);
    free(aligned_b);
}

void mm_kernel2(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel // TODO: try different kernel sizes
    int height = 4;
    int width = 4;
    // TODO: Pad Matrices so that dimensions are multiples of 4 and shared dim is multiple of 32

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows; i += height) { // TODO: use a_rows_padded here
        for (int j = 0; j < b_cols; j += width) { // TODO: use b_cols_padded here
//            std::cout << "i: " << i << " j: " << j << std::endl;
//            kernel2(aligned_a, aligned_b, c, a_rows, b_cols, a_cols, i, j, height, width, 0, a_cols);
            kernel4(aligned_a, aligned_b, c, 0, a_rows, b_cols, a_cols, i, j, height, width, 0, a_cols);

            // for now l=0 and r=a_cols, later in block matrix multiplication, l and r will be updated
        }
    }

    // Clean up
    free(aligned_a);
    free(aligned_b);
}


// Block matrix multiplication
// Want B in L1 Cache because it is accessed in inner loop
// A in L2 Cache, B in L3 Cache again
void mm_blocked(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel size
    int height = 4;
    int width = 4;

    // Block matrix multiplication
    const int b3 = 32;
    const int b2 = 64;
    const int b1 = 128; // TODO: try different block sizes

    for (int i3 = 0; i3 < b_cols; i3 += b3) {
        for (int i2 = 0; i2 < a_cols; i2 += b2) {
            for (int i1 = 0; i1 < a_rows; i1 += b1) {
               // multiply the block
                for (int i = i2; i < std::min(i2+b2, a_cols); i += height) {
                     for (int j = i3; j < std::min(i3 + b3, b_cols); j += width) {
                          kernel(aligned_a, aligned_b, c, a_rows, b_cols, a_cols, i, j, height, width, i1, std::min(i1+b1, a_rows));
                     }
                }
            }
        }
    }
    // Clean up
    free(aligned_a);
    free(aligned_b);
}


// Preprocessing/packing function to align matrices A and B and transpose B
void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned_transposed, const int a_rows, const int a_cols, const int b_cols) {
    // Allocate aligned memory for matrices A and B
    a_aligned.resize(a_rows * a_cols);
    b_aligned_transposed.resize(a_cols * b_cols);

    // Copy data from original matrix A to aligned memory
    std::memcpy(a_aligned.data(), a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            b_aligned_transposed[i * b_cols + j] = b[j * a_cols + i];
        }
    }
}

// Blocked matrix multiplication with packed data
void mm_blocked_packed_stdmin(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_transposed;
    pack(a, b, a_aligned, b_aligned_transposed, a_rows, a_cols, b_cols);

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel size
    int height = 4;
    int width = 4;

    // Block matrix multiplication
    const int b3 = 32;
    const int b2 = 64;
    const int b1 = 128; // TODO: try different block sizes

    for (int i3 = 0; i3 < b_cols; i3 += b3) {
        for (int i2 = 0; i2 < a_cols; i2 += b2) {
            for (int i1 = 0; i1 < a_rows; i1 += b1) {
                // multiply the block
                for (int i = i2; i < std::min(i2+b2, a_cols); i += height) {
                    for (int j = i3; j < std::min(i3 + b3, b_cols); j += width) {
                        kernel(a_aligned.data(), b_aligned_transposed.data(), c, a_rows, b_cols, a_cols, i, j, height, width, i1, std::min(i1+b1, a_rows));
                    }
                }
            }
        }
    }
}

// Blocked matrix multiplication with packed data and no std::min
void mm_blocked_packed(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_transposed;
    pack(a, b, a_aligned, b_aligned_transposed, a_rows, a_cols, b_cols);

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel size
    int height = 4;
    int width = 4;

    // Block matrix multiplication
    const int b3 = 32;
    const int b2 = 64;
    const int b1 = 128; // TODO: try different block sizes

    for (int i3 = 0; i3 < b_cols; i3 += b3) {
        for (int i2 = 0; i2 < a_cols; i2 += b2) {
            for (int i1 = 0; i1 < a_rows; i1 += b1) {
                // multiply the block
//                for (int i = i2; i < std::min(i2+b2, a_cols); i += height) { // no std::min
                 for (int i = i2; i < ((i2+b2) < a_cols ? i2+b2 : a_cols); i += height) {
//                    for (int j = i3; j < std::min(i3 + b3, b_cols); j += width) {
                     for (int j = i3; j < ((i3 + b3) < b_cols ? i3 + b3 : b_cols); j += width) {
                        kernel(a_aligned.data(), b_aligned_transposed.data(), c, a_rows, b_cols, a_cols, i, j, height, width, i1, ((i1+b1) < a_rows ? i1+b1 : a_rows)); // no std::min
                    }
                }
            }
        }
    }
}


