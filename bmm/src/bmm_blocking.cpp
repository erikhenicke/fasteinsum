#include <algorithm>
#include "bmm_blocking.h"

using namespace std;

void blocked_matrix_multiply(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size) {
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; i += block_size) {
            for (int j = 0; j < b_cols; j += block_size) {
                for (int k = 0; k < a_cols; k += block_size) {
                    // Compute block sub-matrix multiplication
                    for (int ii = i; ii < std::min(i + block_size, a_rows); ++ii) {
                        for (int jj = j; jj < std::min(j + block_size, b_cols); ++jj) {
                            for (int kk = k; kk < std::min(k + block_size, a_cols); ++kk) {
                                c[d*a_rows*b_cols + ii*b_cols + jj] += a[d*a_rows*a_cols + ii*a_cols + kk] * b[d*a_cols*b_cols + kk*b_cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}