#include <omp.h>
#include "bmm_omp.h"

double *bmm_collaps(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
#pragma omp parallel for collapse(4)
    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int j = 0; j < b_cols; ++j)
                for (int k = 0; k < a_cols; ++k)
                    c[d*a_rows*b_cols + i*b_cols + j] += a[d*a_rows*a_cols + i*a_cols + k] * b[d*a_cols*b_cols + k*b_cols + j];
    return c;
}

double *bmm_simd(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
#pragma omp parallel for simd
    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int j = 0; j < b_cols; ++j)
                for (int k = 0; k < a_cols; ++k)
                    c[d*a_rows*b_cols + i*b_cols + j] += a[d*a_rows*a_cols + i*a_cols + k] * b[d*a_cols*b_cols + k*b_cols + j];
    return c;
}

double *bmm_omp_simple(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
#pragma omp parallel for
    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int j = 0; j < b_cols; ++j)
                for (int k = 0; k < a_cols; ++k)
                    c[d*a_rows*b_cols + i*b_cols + j] += a[d*a_rows*a_cols + i*a_cols + k] * b[d*a_cols*b_cols + k*b_cols + j];
    return c;
}
