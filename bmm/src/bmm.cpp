#include "bmm.h"

double *batch_matrix_multiply(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int j = 0; j < b_cols; ++j)
                for (int k = 0; k < a_cols; ++k)
                    c[d*a_rows*b_cols + i*b_cols + j] += a[d*a_rows*a_cols + i*a_cols + k] * b[d*a_cols*b_cols + k*b_cols + j];
    return c;
}