#pragma once

double *bmm_naive(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);

void bmm(const double *a, const double *b, double *c, const int bd, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_);