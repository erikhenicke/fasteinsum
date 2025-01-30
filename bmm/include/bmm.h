#pragma once

// naive approach
double *batch_matrix_multiply(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

// optimized approach
void bmm(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols,
         const int h = 8, const int w = 16, const int b1 = 128, const int b2 = 64, const int b3 = 32);