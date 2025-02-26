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
#include <cblas.h>


// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

using namespace std;

void bmm_naive(const double *a, const double *b, double *c,
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


void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int d, const int a_rows, const int b_cols,
            const int a_cols, int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) {

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
                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols), h, w, simd_length, wl);
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

// Version 2: bmm with pipelining kernel "kernel2"
void pack2(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int bd, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int a_cols_padded, const int b_cols_padded) {
    // transposes B and pads A and B

      // Allocate aligned memory for matrices A, B, and C with padded dimensions
    a_aligned.resize(bd * a_rows_padded * a_cols_padded);
    b_aligned.resize(bd * a_cols_padded * b_cols_padded);
    c_aligned.resize(bd * a_rows_padded * b_cols_padded);

   // Copy data from original matrix A to aligned memory and set padded elements to 0
	for (int d = 0; d < bd; ++d) {
    	for (int i = 0; i < a_rows_padded; ++i) {
        	if (i < a_rows) {
            	std::memcpy(&a_aligned[(d * a_rows_padded + i) * a_cols_padded], &a[(d * a_rows + i) * a_cols], a_cols * sizeof(double));
            	std::fill(&a_aligned[(d * a_rows_padded + i) * a_cols_padded + a_cols], &a_aligned[(d * a_rows_padded + (i + 1)) * a_cols_padded], 0.0);
        	} else {
	            std::fill(&a_aligned[(d * a_rows_padded + i) * a_cols_padded], &a_aligned[(d * a_rows_padded + (i + 1)) * a_cols_padded], 0.0);
    	    }
    	}
	}

	// Copy data from original matrix B to aligned memory and set padded elements to 0
    // Transpose simultaniously
	for (int d = 0; d < bd; ++d) {
	    for (int i = 0; i < a_cols_padded; ++i) {
	        for (int j = 0; j < b_cols_padded; ++j) {
	            if (i < a_cols && j < b_cols) {
	                b_aligned[d * a_cols_padded * b_cols_padded + j * a_cols_padded + i] = b[d * a_cols * b_cols + i * b_cols + j];
	            } else {
	                b_aligned[d * a_cols_padded * b_cols_padded + j * a_cols_padded + i] = 0.0;
	            }
	        }
	    }
	}

    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}


// NO PADDING FOR A_COLS (shared dimension)
//    // Copy data from original matrix A to aligned memory and set padded elements to 0
//    for (int d = 0; d < bd; ++d) {
//        for (int i = 0; i < a_rows_padded; ++i) {
//            if (i < a_rows) {
//                std::memcpy(&a_aligned[(d * a_rows_padded + i) * a_cols], &a[(d * a_rows + i) * a_cols], a_cols * sizeof(double));
//            } else {
//                std::fill(&a_aligned[(d * a_rows_padded + i) * a_cols], &a_aligned[(d * a_rows_padded + (i + 1)) * a_cols], 0.0);
//            }
//        }
//    }
//
//    // Copy data from original matrix B to aligned memory, while transposing, and set padded elements to 0
//    for (int d = 0; d < bd; ++d) {
//        for (int i = 0; i < a_cols; ++i) {
//            if (i < a_cols) {
//                for (int j = 0; j < b_cols; ++j) {
//                    b_aligned[d * a_cols * b_cols_padded + j * a_cols + i] = b[d * a_cols * b_cols + i * b_cols + j];
//                }
//                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols + b_cols], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols], 0.0);
//            } else {
//                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols], 0.0);
//            }
//        }
//    }



void kernel2(double *aligned_a, double *aligned_b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols,
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

void bmm2(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1_, int b2_, int b3_,
    void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int, int, int)) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;
    int a_cols_padded = a_cols + (32 - a_cols % 32) % 32; //pad to 32 for kernel2

    // Block sizes need to be multiples of h and w
    int b1 = b1_ - (b1_ % 32); // b1 needs to be multiple of 32 for kernel2
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_transposed;
    aligned_vector<double> c_aligned;
    pack2(a, b, a_aligned, b_aligned_transposed, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, a_cols_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
                            kernel(a_aligned.data(), b_aligned_transposed.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols_padded, k, j, i1, i1 + b1, h, w); // std::min(i1 + b1, a_cols) not needed because a_cols_padded % 32 = 0
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


// bmm with blocking, OMP SIMD parallel directive (should work on every architecture, no AVX intrinsics)
void bmm3(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int b1, const int b2, const int b3) {
    // fill c with 0
    std::fill(c, c + batch_dim * a_rows * b_cols, 0.0);
    // TODO: Transpose b?

    #pragma omp parallel for simd collapse(4)
//#pragma omp parallel for shared(matrixA, matrixB, matrixC) schedule(static) num_threads(THREADS)
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < b_cols; i += b3) {
            for (int j = 0; j < a_rows; j += b2) {
                for (int k = 0; k < a_cols; k += b1) {
                    // Compute block sub-matrix multiplication
                    for (int ii = i; ii < std::min(i + b3, b_cols); ++ii) {
                        for (int jj = j; jj < std::min(j + b2, a_rows); ++jj) {
                            for (int kk = k; kk < std::min(k + b1, a_cols); ++kk) {
                                c[d*a_rows*b_cols + jj*b_cols + ii] += a[d*a_rows*a_cols + jj*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + ii];
                            }
                        }
                    }
                }
            }
        }
    }
}

void bmm_blas(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
    // fill c with 0
    std::fill(c, c + batch_dim * a_rows * b_cols, 0.0);

    int sizeA = a_rows * a_cols;
    int sizeB = a_cols * b_cols;
    int sizeC = a_rows * b_cols;

    // Perform matrix multiplication using BLAS
    #pragma omp parallel for
    for (int d = 0; d < batch_dim; ++d) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, a_cols, 1.0, &a[d * sizeA], a_cols, &b[d * sizeB], b_cols, 1.0, &c[d * sizeC], b_cols);
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
    const int b_cols = 512;
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

    for (int i = 0; i < 1; ++i) {
        bmm(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, h, w, simd_length, wl, b1, b2_, b3_);
    }
    // Call bmm

    cout << "Finished bmm" << endl;

    //     Compare this with the reference implementation
     aligned_vector<double> c_ref(bd * a_rows * b_cols, 0.0);
     for (int d = 0; d < bd; ++d) {
         bmm_naive(&a[d * a_rows * a_cols], &b[d * a_cols * b_cols], &c_ref[d * a_rows * b_cols], 1, a_rows, b_cols, a_cols);
     }

     // Check if the results are correct
     bool equal = true;
     for (int i = 0; i < bd * a_rows * b_cols; ++i) {
         if (abs(c[i] - c_ref[i]) > 1e-6) {
             equal = false;
             break;
         }
     }

     cout << "Result of bmm is " << (equal ? "correct" : "incorrect") << endl;

    cout << "Starting bmm2" << endl;

    for (int i = 0; i < 1; ++i) {
        bmm2(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, h, w, 0, 0, b1, b2_, b3_, kernel2);
    }
    // Call bmm

    cout << "Finished bmm2" << endl;

//     Compare this with the reference implementation

     // Check if the results are correct
     equal = true;
     for (int i = 0; i < bd * a_rows * b_cols; ++i) {
         if (abs(c[i] - c_ref[i]) > 1e-6) {
             equal = false;
             break;
         }
     }

     cout << "Result of bmm2 is " << (equal ? "correct" : "incorrect") << endl;


    cout << "Starting bmm3" << endl;

    for (int i = 0; i < 1; ++i) {
        bmm3(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, b1, b2_, b3_);
    }

    cout << "Finished bmm3" << endl;

    //     Compare this with the reference implementation

        // Check if the results are correct
        equal = true;
        for (int i = 0; i < bd * a_rows * b_cols; ++i) {
            if (abs(c[i] - c_ref[i]) > 1e-6) {
                equal = false;
                break;
            }
        }

        cout << "Result of bmm3 is " << (equal ? "correct" : "incorrect") << endl;

    cout << "Starting bmm_blas" << endl;

    for (int i = 0; i < 1; ++i) {
        bmm_blas(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols);
    }

    cout << "Finished bmm_blas" << endl;

    //     Compare this with the reference implementation

    // Check if the results are correct
    equal = true;
    for (int i = 0; i < bd * a_rows * b_cols; ++i) {
        if (abs(c[i] - c_ref[i]) > 1e-6) {
            equal = false;
            break;
        }
    }

    cout << "Result of bmm_blas is " << (equal ? "correct" : "incorrect") << endl;

    return 0;
}



//
//// test pack2
//int main() {
//    // Define small matrix dimensions
//    const int bd = 1;
//    const int a_rows = 3;
//    const int b_cols = 7;
//    const int a_cols = 5;
//    const int h = 2;
//    const int w = 2;
//    const int b1 = 2;
//    const int b2_ = 2;
//    const int b3_ = 2;
//
//    // Allocate memory for small matrices A and B
//    aligned_vector<double> a(bd * a_rows * a_cols);
//    aligned_vector<double> b(bd * a_cols * b_cols);
//    aligned_vector<double> c(bd * a_rows * b_cols, 0.0);
//
//    // Initialize small matrices A and B
//    for (int i = 0; i < bd * a_rows * a_cols; ++i) {
//        a[i] = i + 1;
//    }
//    for (int i = 0; i < bd * a_cols * b_cols; ++i) {
//        b[i] = i + 1;
//    }
//
//    // padding
//    int pad_a_cols = 10;
//    int a_rows_padded = a_rows + (h - a_rows % h) % h;
//    int b_cols_padded = b_cols + (w - b_cols % w) % w;
//    int a_cols_padded = a_cols + (pad_a_cols - a_cols % pad_a_cols) % pad_a_cols; //pad to 32 for kernel2
//
//    // Pack data
//    aligned_vector<double> a_aligned;
//    aligned_vector<double> b_aligned_transposed;
//    aligned_vector<double> c_aligned;
//    pack2(a.data(), b.data(), a_aligned, b_aligned_transposed, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, a_cols_padded, b_cols_padded);
//
//    // Print original matrices
//    std::cout << "Matrix A:" << std::endl;
//    for (int i = 0; i < a.size(); ++i) {
//        std::cout << a[i] << " ";
//        if ((i + 1) % a_cols == 0) std::cout << std::endl;
//    }
//
//    std::cout << "Matrix B:" << std::endl;
//    for (int i = 0; i < b.size(); ++i) {
//        std::cout << b[i] << " ";
//        if ((i + 1) % b_cols == 0) std::cout << std::endl;
//    }
//
//    // Print packed matrices
//    std::cout << "Packed matrix A:" << std::endl;
//    for (int i = 0; i < a_aligned.size(); ++i) {
//        std::cout << a_aligned[i] << " ";
//        if ((i + 1) % a_cols_padded == 0) std::cout << std::endl;
//    }
//
//    std::cout << "Packed and transposed matrix B:" << std::endl;
//    for (int i = 0; i < b_aligned_transposed.size(); ++i) {
//        std::cout << b_aligned_transposed[i] << " ";
//        if ((i + 1) % a_cols_padded == 0) std::cout << std::endl;
//    }
//
//    return 0;
//}