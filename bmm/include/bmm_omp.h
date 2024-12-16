#pragma once

double *bmm_collaps(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);//

double *bmm_simd(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);//

double *bmm_omp_simple(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols);//
