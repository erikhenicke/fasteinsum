#include "bmm.h"
#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <random>
//#include <chrono>
//#include <fstream>
#include <iostream>


// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

using namespace std;

double *bmm_naive(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {

    std::fill(c, c + batch_dim * a_rows * b_cols, 0.0);

    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int k = 0; k < a_cols; ++k) {
                for (int j = 0; j < b_cols; ++j) {
                    // Initialization
                    // c[d * a_rows * b_cols + i * b_cols + j] = (k != 0) * c[d * a_rows * b_cols + i * b_cols + j];
                    // if (i == 0 && j == 0) {
                    //     std::cout << "AA[" << i << "," << k << "] = " << a[d * a_rows * a_cols + i * a_cols + k] << std::endl;
                    //     std::cout << "BB[" << k << "," << j << "] = " << b[d * a_cols * b_cols + j * b_cols + k] << std::endl;
                    //     std::cout << "AA[" << i << "," << k << "] * BB[" << k << "," << j << "] = " << a[d * a_rows * a_cols + i * a_cols + k] * b[d * a_cols * b_cols + j * b_cols + k] << std::endl;
                    // }
                    c[(d * a_rows + i) * b_cols + j] += a[(d * a_rows + i) * a_cols + k] * b[(d * a_cols + k) * b_cols + j];
                }
            }
    return c;
}


// Matrix multiplication for batched matrices with kernel, blocking, and SIMD optimizations
// a: input matrix A
// b: input matrix B
// c: output matrix C
// batch_dim: number of matrices in batch
// a_rows: number of rows in matrix A
// b_cols: number of columns in matrix B
// a_cols: number of columns in matrix A and rows in matrix B

// Later: Parallelization with OpenMP (how exactly? (!) Kernel updates global variable c)
// Second kernel without SIMD for other architectures
// Optimization of block sizes and kernel sizes: hardcode and unroll loops. Now, variable sizes
// Try restrict keyword for pointers
// Update c in kernel with vector intrinsics

void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded) {
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

    // Copy data from original matrix B to aligned memory and set padded elements to 0
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_cols; ++i) {
            if (i < a_cols) {
                std::memcpy(&b_aligned[(d * a_cols + i) * b_cols_padded], &b[(d * a_cols + i) * b_cols], b_cols * sizeof(double));
                std::fill(&b_aligned[(d * a_cols + i) * b_cols_padded + b_cols], &b_aligned[(d * a_cols + (i + 1)) * b_cols_padded], 0.0);
            } else {
                std::fill(&b_aligned[(d * a_cols + i) * b_cols_padded], &b_aligned[(d * a_cols + (i + 1)) * b_cols_padded], 0.0);
            }
        }
    }

    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}


void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) {

    // Create a vector of __m256d with size height * width
    aligned_vector<__m256d> t(h * wl); // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
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

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    //    int b1 = b1_ - (b1_ % simd_length);
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    #pragma omp parallel for // collapse(4) // TODO: update c right always???
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols) , h, w, simd_length, wl);
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

void generate_random_matrix(aligned_vector<double> &matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// write main function that calls bmm
int main() {
    // Define matrix dimensions
    const int bd = 4;
    const int a_rows = 256;
    const int b_cols = 256;
    const int a_cols = 256;
    const int h = 6;
    const int w = 8;
    const int simd_length = 4;
    const int wl = w / simd_length;
    const int b1 = 32;
    const int b2_ = 64;
    const int b3_ = 128;

    // Allocate memory for matrices A, B, and C
    aligned_vector<double> a(bd * a_rows * a_cols);
    aligned_vector<double> b(bd * a_cols * b_cols);
    aligned_vector<double> c(bd * a_rows * b_cols, 0.0);

    // Generate random matrices A and B
    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    cout << "Starting bmm" << endl;

    for (int i = 0; i < 100; ++i) {
        bmm(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, h, w, simd_length, wl, b1, b2_, b3_);
    }
    // Call bmm

    cout << "Finished bmm" << endl;

    // Compare this with the reference implementation
    // aligned_vector<double> c_ref(bd * a_rows * b_cols, 0.0);
    // for (int d = 0; d < bd; ++d) {
    //     bmm_naive(&a[d * a_rows * a_cols], &b[d * a_cols * b_cols], &c_ref[d * a_rows * b_cols], 1, a_rows, b_cols, a_cols);
    // }

    // // Check if the results are correct
    // bool equal = true;
    // for (int i = 0; i < bd * a_rows * b_cols; ++i) {
    //     if (abs(c[i] - c_ref[i]) > 1e-6) {
    //         equal = false;
    //         break;
    //     }
    // }

    // cout << "Result of bmm is " << (equal ? "correct" : "incorrect") << endl;

    return 0;
}

