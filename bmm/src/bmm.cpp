#include "bmm.h"
#include "kernels.h"
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

// TODO: update summary once these functions are implemented
// Final BMM versions:
// 1. Naive BMM
// 2. Blocked BMM (with Transposition)
// 3. Blocked BMM (with Transposition) and Parallelization and Vectorization using OMP
// 4. As above but using a hardware specific kernel with vector intrinsics/(AVX2) instructions
// 5. ?? Using kernel but with only OMP SIMD
// 6. BLAS BMM

// Other functions:
// Preprocessing functions: pack, packT, pack_T_pad
// Other BMM versions:
//     - bmm_simple_kernel (using simple kernel)
//     - bmm_var_kernel (using kernel with variable size, no loop unrolling)



/**
 * \brief Pack data into aligned memory.
 *
 * This function packs the data of matrices A and B into aligned memory for efficient memory access.
 * The data is padded (e.g. to be multiples of h and w) and C is initialized with zeros.
 *
 * \param a Pointer to matrix A.
 * \param b Pointer to matrix B.
 * \param a_aligned Aligned memory for matrix A.
 * \param b_aligned Aligned memory for matrix B.
 * \param c_aligned Aligned memory for matrix C.
 * \param bd Batch dimension.
 * \param a_rows Number of rows of matrix A.
 * \param a_cols Number of columns of matrix A.
 * \param b_cols Number of columns of matrix B.
 * \param a_rows_padded Number of rows of matrix A after padding.
 * \param b_cols_padded Number of columns of matrix B after padding.
 */
void pack(
    const double *a,
    const double *b,
    aligned_vector<double> &a_aligned,
    aligned_vector<double> &b_aligned,
    aligned_vector<double> &c_aligned,
    const int bd, const int a_rows,
    const int a_cols, const int b_cols,
    const int a_rows_padded,
    const int b_cols_padded)
{
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

/**
 * \brief Pack data into aligned memory and transpose B.
 *
 * This function packs the data of matrices A and B into aligned memory for efficient memory access.
 * The data is padded (e.g. to be multiples of h and w) and C is initialized with zeros.
 * The matrix B is transposed.
 *
 * \param a Pointer to matrix A.
 * \param b Pointer to matrix B.
 * \param a_aligned Aligned memory for matrix A.
 * \param b_aligned Aligned memory for matrix B.
 * \param c_aligned Aligned memory for matrix C.
 * \param bd Batch dimension.
 * \param a_rows Number of rows of matrix A.
 * \param a_cols Number of columns of matrix A.
 * \param b_cols Number of columns of matrix B.
 * \param a_rows_padded Number of rows of matrix A after padding.
 * \param b_cols_padded Number of columns of matrix B after padding.
 */
void packT(
    const double *a,
    const double *b,
    aligned_vector<double> &a_aligned,
    aligned_vector<double> &b_aligned,
    aligned_vector<double> &c_aligned,
    const int bd, const int a_rows,
    const int a_cols, const int b_cols,
    const int a_rows_padded,
    const int b_cols_padded)
{
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
    // Transpose simultaniously
    for (int d = 0; d < bd; ++d) {
        for (int i = 0; i < a_cols; ++i) {
            for (int j = 0; j < b_cols_padded; ++j) {
                if (i < a_cols && j < b_cols) {
                    b_aligned[d * a_cols * b_cols_padded + j * a_cols + i] = b[d * a_cols * b_cols + i * b_cols + j];
                } else {
                    b_aligned[d * a_cols * b_cols_padded + j * a_cols + i] = 0.0;
                }
            }
        }
    }

    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}


/**
* \brief Pack data into aligned memory with transposition.
*
* Same as "pack" but additionally transposes matrix B.
* Also padding for a_cols.
* Used for simple kernel, there padding needed for a_cols.
*/
void pack_T_pad(
    const double *a,
    const double *b,
    aligned_vector<double> &a_aligned,
    aligned_vector<double> &b_aligned,
    aligned_vector<double> &c_aligned,
    const int bd, const int a_rows,
    const int a_cols,
    const int b_cols,
    const int a_rows_padded,
    const int a_cols_padded,
    const int b_cols_padded)
{
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

inline void packB(const double *b, aligned_vector<double> &b_pack, const int bd_idx, const int rows_idx, int &rows_idx_bound,
           const int cols_idx, int &cols_idx_bound, const int block_height, const int block_width, const int b_rows,
           const int b_cols, const int b_cols_padded) {
    if (cols_idx + block_width < b_cols_padded) {
        cols_idx_bound = cols_idx + block_width;
    } else {
        cols_idx_bound = b_cols_padded;
    }

    if (rows_idx + block_height < b_rows) {
        rows_idx_bound = rows_idx + block_height;
    } else {
        rows_idx_bound = b_rows;
    }

    // Copy data from matrix B to aligned memory and padd with zeros if necessary.
    int b_pack_rows = rows_idx_bound - rows_idx;
    int b_pack_cols = cols_idx_bound - cols_idx;
    if (cols_idx_bound > b_cols) {
        for (int i = 0; i < b_pack_rows; ++i) {
            std::memcpy(&b_pack[b_pack_cols * i], &b[(bd_idx * b_rows + rows_idx + i) * b_cols + cols_idx],
                        (b_cols - cols_idx) * sizeof(double));
            std::fill(&b_pack[b_pack_cols * i + b_cols - cols_idx], &b_pack[b_pack_cols * (i + 1)], 0.0);
        }
    } else {
        for (int i = 0; i < b_pack_rows; ++i) {
            std::memcpy(&b_pack[b_pack_cols * i], &b[(bd_idx * b_rows + rows_idx + i) * b_cols + cols_idx],
                        b_pack_cols * sizeof(double));
        }
    }
}

inline void packA(const double *a, aligned_vector<double> &a_pack, const int bd_idx, const int rows_idx, int &rows_idx_bound, const int cols_idx,
           int cols_idx_bound, const int block_height, const int a_rows, const int a_cols, const int a_rows_padded) {
    if (rows_idx + block_height < a_rows_padded) {
        rows_idx_bound = rows_idx + block_height;
    } else {
        rows_idx_bound = a_rows_padded;
    }

    // Copy data from matrix A to aligned memory and padd with zeros if necessary.
    int a_pack_rows = rows_idx_bound - rows_idx;
    int a_pack_cols = cols_idx_bound - cols_idx;
    if (rows_idx_bound > a_rows) {
        for (int i = 0; i < a_rows - rows_idx; ++i) {
            std::memcpy(&a_pack[a_pack_cols * i], &a[(bd_idx * a_rows + rows_idx + i) * a_cols + cols_idx],
                        a_pack_cols * sizeof(double));
        }
        for (int i = a_rows - rows_idx; i < a_pack_rows; ++i) {
            std::fill(&a_pack[a_pack_cols * i], &a_pack[a_pack_cols * (i + 1)], 0.0);
        }
    } else {
        for (int i = 0; i < a_pack_rows; ++i) {
            std::memcpy(&a_pack[a_pack_cols * i], &a[(bd_idx * a_rows + rows_idx + i) * a_cols + cols_idx],
                        a_pack_cols * sizeof(double));
        }
    }
}

/**
 * \brief Naive batch multiplication of two matrices.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C
 * using four nested loops. The matrices may have a batch dimension.
 *
 * \param[in] a Pointer to the first input matrix A.
 * \param[in] b Pointer to the second input matrix B.
 * \param[out] c Pointer to the output matrix C.
 * \param[in] batch_dim Number of matrices in the batch.
 * \param[in] a_rows Number of rows in matrix A.
 * \param[in] b_cols Number of columns in matrix B.
 * \param[in] a_cols Number of columns in matrix A and rows in matrix B.
 */
void bmm_naive(
    const double *a,
    const double *b,
    double *c,
    const int batch_dim,
    const int a_rows,
    const int b_cols,
    const int a_cols)
{
    std::fill(c, c + batch_dim * a_rows * b_cols, 0.0);

    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            for (int k = 0; k < a_cols; ++k) {
                for (int j = 0; j < b_cols; ++j) {
                    c[(d * a_rows + i) * b_cols + j] += a[(d * a_rows + i) * a_cols + k] * b[(d * a_cols + k) * b_cols + j];
                }
            }
    	}
    }
}

/**
 * \brief Batch multiplication of two matrices with transposition, blocking and parallelization.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C
 * using blocking, transposition, and parallelization with OpenMP.
 *
 * \param[in] a Pointer to the first input matrix A.
 * \param[in] b Pointer to the second input matrix B.
 * \param[out] c Pointer to the output matrix C.
 * \param[in] batch_dim Number of matrices in the batch.
 * \param[in] a_rows Number of rows in matrix A.
 * \param[in] b_cols Number of columns in matrix B.
 * \param[in] a_cols Number of columns in matrix A and rows in matrix B.
 * \param[in] b1 Block size in the first dimension.
 * \param[in] b2 Block size in the second dimension.
 * \param[in] b3 Block size in the third dimension.
 */
void bmm_T_bl_para(
    const double *a,
    const double *b,
    double *c,
    const int batch_dim,
    const int a_rows,
    const int b_cols,
    const int a_cols,
    const int b1,
    const int b2,
    const int b3)
{
    // fill c with 0
    std::fill(c, c + batch_dim * a_rows * b_cols, 0.0);
    // TODO: Transpose b?

    #pragma omp parallel for simd collapse(3)
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

/**
 * \brief Batch multiplication of two matrices using BLAS.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C
 * using the BLAS library general matrix multiplication function for doubles. The matrices may have a batch dimension.
 *
 * \param[in] a Pointer to the first input matrix A.
 * \param[in] b Pointer to the second input matrix B.
 * \param[out] c Pointer to the output matrix C.
 * \param[in] batch_dim Number of matrices in the batch.
 * \param[in] a_rows Number of rows in matrix A.
 * \param[in] b_cols Number of columns in matrix B.
 * \param[in] a_cols Number of columns in matrix A and rows in matrix B.
 */
void bmm_blas(
    const double *a,
    const double *b,
    double *c,
    const int batch_dim,
    const int a_rows,
    const int b_cols,
    const int a_cols)
{
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



/**
 * \brief BMM with variable kernel size.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C
 * using a kernel with variable block sizes. The matrices may have a batch dimension.
 *
 * \param a Pointer to the first input matrix A.
 * \param b Pointer to the second input matrix B.
 * \param c Pointer to the output matrix C.
 * \param bd Batch dimension.
 * \param a_rows_padded Number of rows of matrix A after padding.
 * \param b_cols_padded Number of columns of matrix B after padding.
 * \param a_cols Number of columns of matrix A and rows of matrix B.
 * \param h Height of the block.
 * \param w Width of the block.
 * \param simd_length SIMD length.
 * \param wl w / simd_length.
 * \param b1 Block size in the first dimension.
 * \param b2 Block size in the second dimension (may vary sligtly to be compatible with h).
 * \param b3 Block size in the third dimension (may vary slightly to be compatible with w).
 */
void bmm_var_kernel(
    const double *a,
    const double *b,
    double *c,
    const int bd,
    const int a_rows,
    const int b_cols,
    const int a_cols,
    int h,
    int w,
    int simd_length,
    int wl,
    int b1,
    int b2_,
    int b3_)
{
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
                            kernel_var(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols), h, w, simd_length, wl);
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

/**
 * \brief Batch matrix multiplication using a simple kernel with variable block sizes.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C
 * using a kernel with variable block sizes. The matrices may have a batch dimension.
 * The function uses AVX intrinsics with pipelined FMA calls for efficient computation.
 *
 * \param[in] a Pointer to the first input matrix A.
 * \param[in] b Pointer to the second input matrix B.
 * \param[out] c Pointer to the output matrix C.
 * \param[in] bd Batch dimension.
 * \param[in] a_rows Number of rows in matrix A.
 * \param[in] b_cols Number of columns in matrix B.
 * \param[in] a_cols Number of columns in matrix A and rows in matrix B.
 * \param[in] h Height of the block.
 * \param[in] w Width of the block.
 * \param[in] simd_length SIMD length.
 * \param[in] wl Width divided by SIMD length.
 * \param[in] b1_ Block size in the first dimension (will be adjusted to be a multiple of 32).
 * \param[in] b2_ Block size in the second dimension (will be adjusted to be a multiple of h).
 * \param[in] b3_ Block size in the third dimension (will be adjusted to be a multiple of w).
 * \param[in] kernel Pointer to the kernel function used for block matrix multiplication. -> use kernel2
 */
void bmm_simple_kernel(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1_, int b2_, int b3_,
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
    pack_T_pad(a, b, a_aligned, b_aligned_transposed, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, a_cols_padded, b_cols_padded);

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


// bmm versions with different parallelization and vectorization strategies (kernel parameter)

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
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
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

void bmm_parallelT(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_,
         void (*kernelT)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)) {

    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_trans;
    aligned_vector<double> c_aligned;
    packT(a, b, a_aligned, b_aligned_trans, c_aligned, bd, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
#pragma omp parallel for collapse(3)
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_cols; i1 += b1) {
                    for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                        for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
                            kernelT(a_aligned.data(), b_aligned_trans.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols));
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

void bmm_pack(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols,
                    const int a_cols, int h, int w, int b1, int b2_, int b3_) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
    // int b1 = b1_ - (b1_ % simd_length);
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> c_aligned;
    c_aligned.resize(bd * a_rows_padded * b_cols_padded);
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int d = 0; d < bd; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i1 = 0; i1 < a_cols; i1 += b1) {
                // Init packed A~
                aligned_vector<double> a_pack;
                a_pack.resize(b2 * b1);
                // pack B to B~
                int J, R;
                aligned_vector<double> b_pack;
                b_pack.resize(b1 * b3);
                packB(b, b_pack, d, i1, R, i3, J, b1, b3, a_cols, b_cols, b_cols_padded);
                for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
                    // pack A to A~
                    int K;
                    packA(a, a_pack, d, i2, K, i1, R, b2, a_rows, a_cols, a_rows_padded);
                    for (int k = i2; k < K; k += h) {
                        for (int j = i3; j < J; j += w) {
                            kernel_8x16_pack(a_pack.data(), b_pack.data(), c_aligned.data(), d, a_rows_padded,
                                               b_cols_padded, (J - i3), k, j, k - i2, j - i3, i1, R);
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
            std::memcpy(&c[offsetC + i * b_cols], &c_aligned[offsetCAligned + i * b_cols_padded],
                        b_cols * sizeof(double));
        }
    }
}

