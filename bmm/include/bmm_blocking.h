// File: bmm_blocking.h

#ifndef BMM_BLOCKING_H
#define BMM_BLOCKING_H

void blocked_matrix_multiply(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols, const int block_size = 32);

#endif // BMM_BLOCKING_H