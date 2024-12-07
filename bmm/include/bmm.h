#pragma once

double *batch_matrix_multiply(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);