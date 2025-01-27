//
// Created by sonja on 26.01.25.
//

#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include "aligned_allocator.h"

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

void kernel(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int l, int r) ;
void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols);

#endif //KERNEL_H
