//#include <algorithm>
#include "bmm_blocking.h"
#include <omp.h>


void bmm_blocked(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size) {
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; i += block_size) {
            for (int k = 0; k < a_cols; k += block_size) {
                for (int j = 0; j < b_cols; j += block_size) {
                    // Compute block sub-matrix multiplication (do not use std::min to avoid overhead)
                    for (int ii = i; ii < ((i + block_size < a_rows) ? i + block_size : a_rows); ++ii) {
                        for (int kk = k; kk < ((k + block_size < a_cols) ? k + block_size : a_cols); ++kk) {
                            for (int jj = j; jj < ((j + block_size < b_cols) ? j + block_size : b_cols); ++jj) {
                                c[d*a_rows*b_cols + ii*b_cols + jj] += a[d*a_rows*a_cols + ii*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

void bmm_blocked_simd(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size) {
#pragma omp parallel for simd
    //#pragma omp parallel for shared(matrixA, matrixB, matrixC) schedule(static) num_threads(THREADS)
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; i += block_size) {
            for (int k = 0; k < a_cols; k += block_size) {
                for (int j = 0; j < b_cols; j += block_size) {
                    // Compute block sub-matrix multiplication (do not use std::min to avoid overhead)
                    for (int ii = i; ii < ((i + block_size < a_rows) ? i + block_size : a_rows); ++ii) {
                        for (int kk = k; kk < ((k + block_size < a_cols) ? k + block_size : a_cols); ++kk) {
                            for (int jj = j; jj < ((j + block_size < b_cols) ? j + block_size : b_cols); ++jj) {
                                c[d*a_rows*b_cols + ii*b_cols + jj] += a[d*a_rows*a_cols + ii*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

void bmm_blocked_simd_collapse(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size) {
    #pragma omp parallel for simd collapse(4)
//#pragma omp parallel for shared(matrixA, matrixB, matrixC) schedule(static) num_threads(THREADS)
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; i += block_size) {
            for (int k = 0; k < a_cols; k += block_size) {
                for (int j = 0; j < b_cols; j += block_size) {
                    // Compute block sub-matrix multiplication (do not use std::min to avoid overhead)
                    for (int ii = i; ii < ((i + block_size < a_rows) ? i + block_size : a_rows); ++ii) {
                        for (int kk = k; kk < ((k + block_size < a_cols) ? k + block_size : a_cols); ++kk) {
                            for (int jj = j; jj < ((j + block_size < b_cols) ? j + block_size : b_cols); ++jj) {
                                c[d*a_rows*b_cols + ii*b_cols + jj] += a[d*a_rows*a_cols + ii*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

// for matrix multiplication memory locations are non-overlapping - TODO: check, what if A * A = C etc?
void bmm_blocked_simd_restricted_pointers(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size) {
#pragma omp parallel for simd
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; i += block_size) {
            for (int k = 0; k < a_cols; k += block_size) {
                for (int j = 0; j < b_cols; j += block_size) {
                    // Compute block sub-matrix multiplication (do not use std::min to avoid overhead)
                    for (int ii = i; ii < ((i + block_size < a_rows) ? i + block_size : a_rows); ++ii) {
                        for (int kk = k; kk < ((k + block_size < a_cols) ? k + block_size : a_cols); ++kk) {
                            for (int jj = j; jj < ((j + block_size < b_cols) ? j + block_size : b_cols); ++jj) {
                                c[d*a_rows*b_cols + ii*b_cols + jj] += a[d*a_rows*a_cols + ii*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}



