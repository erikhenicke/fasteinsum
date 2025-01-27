//
// Created by sonja on 26.01.25.
//
#include "kernel.h"
#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
//#include <omp.h>
//#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>


// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;
//using aligned_vector_simd = std::vector<__m256d, aligned_allocator<__m256d, 64>>;

//using namespace std;

const int h = 6; // kernel height
const int w = 8; // kernel width
const int simd_length = 4; // number of doubles in a SIMD register
const int wl = 2; //width divided by simd_length
// TODO: In the end hardcode height and width

// Naive implementation using three nested loops
void mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];
}

// Preprocessing/packing function to align matrices A and B and C
// (!) B is NOT transposed
// c is initialized to zero
void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int a_rows, const int a_cols, const int b_cols, const int a_rows_padded, const int b_cols_padded) {
    // Allocate aligned memory for matrices A, B, and C with padded dimensions
    a_aligned.resize(a_rows_padded * a_cols);
    b_aligned.resize(a_cols * b_cols_padded);
    c_aligned.resize(a_rows_padded * b_cols_padded);

    // Copy data from original matrix A to aligned memory and set padded elements to 0
    for (int i = 0; i < a_rows_padded; ++i) {
        if (i < a_rows) {
            std::memcpy(&a_aligned[i * a_cols], &a[i * a_cols], a_cols * sizeof(double));
        } else {
            std::fill(&a_aligned[i * a_cols], &a_aligned[(i + 1) * a_cols], 0.0);
        }
    }

    // Copy data from original matrix B to aligned memory and set padded elements to 0
    for (int i = 0; i < a_cols; ++i) {
        if (i < a_cols) {
            std::memcpy(&b_aligned[i * b_cols_padded], &b[i * b_cols], b_cols * sizeof(double));
            std::fill(&b_aligned[i * b_cols_padded + b_cols], &b_aligned[(i + 1) * b_cols_padded], 0.0);
        } else {
            std::fill(&b_aligned[i * b_cols_padded], &b_aligned[(i + 1) * b_cols_padded], 0.0);
        }
    }

    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}


//void pack(const double *a, const double *b, const double *c, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
//          aligned_vector<double> &c_aligned, const int a_rows, const int a_cols, const int b_cols) {
//    // Allocate aligned memory for matrices A and B
//    a_aligned.resize(a_rows * a_cols);
//    b_aligned.resize(a_cols * b_cols);
//    c_aligned.resize(a_rows * b_cols);
//
//    // Copy data from original matrix A and B to aligned memory
//    std::memcpy(a_aligned.data(), a, a_rows * a_cols * sizeof(double));
//    std::memcpy(b_aligned.data(), b, a_cols * b_cols * sizeof(double));
//
//    // Initialize result matrix c to zero
//    for (int i = 0; i < a_rows * b_cols; ++i)
//        c_aligned[i] = 0.0;
//
//}



void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows_padded; i += h) {
        for (int j = 0; j < b_cols_padded; j += w) {
            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows_padded, b_cols_padded, a_cols, i, j, 0, a_cols);
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int i = 0; i < a_rows; ++i) {
        std::memcpy(&c[i * b_cols], &c_aligned[i * b_cols_padded], b_cols * sizeof(double));
    }
}

//void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
//    // Pad a_rows and b_cols to be multiples of h and w
//    int a_rows_padded = a_rows + (h - a_rows % h) % h;
//    int b_cols_padded = b_cols + (w - b_cols % w) % w;
//
//    // Pack data
//    aligned_vector<double> a_aligned;
//    aligned_vector<double> b_aligned;
//    aligned_vector<double> c_aligned;
//    pack(a, b, c, a_aligned, b_aligned, c_aligned, a_rows, a_cols, b_cols);
//
//    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
//    for (int i = 0; i < a_rows; i += h) { // TODO: use a_rows_padded here
//        for (int j = 0; j < b_cols; j += w) { // TODO: use b_cols_padded here
////            std::cout << "i: " << i << " j: " << j << std::endl;
//            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows, b_cols, a_cols, i, j, 0, a_cols);
//            // for now l=0 and r=a_cols, later in block matrix multiplication, l and r will be updated
//        }
//    }
//    // Copy data from aligned memory to original matrix C
//    std::memcpy(c, c_aligned.data(), a_rows * b_cols * sizeof(double));
//}

void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r) {

//    std::cout << "0" << std::endl;

    // Create a vector of __m256d with size height * width
    aligned_vector<__m256d> t(h * wl); // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
    // t[i]: i-th element of t (a 4 element _m256d SIMD vector), t[i][j]: j-th double of i-th element of t

    // Initialize each element with zeros
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < h * wl; ++i) {
        t[i] = zero;
    }

    // Print t
//    for (int i = 0; i < h * wl; ++i) {
//        for (int j = 0; j < simd_length; ++j) {
//        	std::cout << t[i][j] << " ";
//        }
//    }

//    std::cout << "1" << std::endl;

    for (int k = l; k < r; k++) {
        __m256d b0 = _mm256_load_pd(&b_aligned[k * b_cols + b_idx]);
        __m256d b1 = _mm256_load_pd(&b_aligned[k * b_cols + b_idx + simd_length]);
//        __m256d b0 = _mm256_load_pd(&b_aligned[(k * b_cols + b_idx)/simd_length]);
//        __m256d b1 = _mm256_load_pd(&b_aligned[(k * b_cols + b_idx)/simd_length + 1]); // indexing???

//        std::cout << "2" << std::endl;

        __m256d a0 = _mm256_broadcast_sd(&a_aligned[a_idx * a_cols + k]);
        t[0] = _mm256_fmadd_pd(a0, b0, t[0]);
        t[1] = _mm256_fmadd_pd(a0, b1, t[1]);

        __m256d a1 = _mm256_broadcast_sd(&a_aligned[(a_idx + 1) * a_cols + k]);
        t[2] = _mm256_fmadd_pd(a1, b0, t[2]);
        t[3] = _mm256_fmadd_pd(a1, b1, t[3]);

        __m256d a2 = _mm256_broadcast_sd(&a_aligned[(a_idx + 2) * a_cols + k]);
        t[4] = _mm256_fmadd_pd(a2, b0, t[4]);
        t[5] = _mm256_fmadd_pd(a2, b1, t[5]);

        __m256d a3 = _mm256_broadcast_sd(&a_aligned[(a_idx + 3) * a_cols + k]);
        t[6] = _mm256_fmadd_pd(a3, b0, t[6]);
        t[7] = _mm256_fmadd_pd(a3, b1, t[7]);

        __m256d a4 = _mm256_broadcast_sd(&a_aligned[(a_idx + 4) * a_cols + k]);
        t[8] = _mm256_fmadd_pd(a4, b0, t[8]);
        t[9] = _mm256_fmadd_pd(a4, b1, t[9]);

        __m256d a5 = _mm256_broadcast_sd(&a_aligned[(a_idx + 5) * a_cols + k]);
        t[10] = _mm256_fmadd_pd(a5, b0, t[10]);
        t[11] = _mm256_fmadd_pd(a5, b1, t[11]);
    }


// TODO: align c

    // Update c with the values in t
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < wl; ++j) {
            _mm256_store_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length], t[i * wl + j]);
        }
    }
}



int main() {
  // mulktiply two random matrices of size 1024 x 1024 using mm_kernel
    int size = 703;
    std::vector<double> a(size * size), b(size * size), c(size * size);
    for (int i = 0; i < size * size; ++i) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // measure the time taken by mm_kernel
    auto start = std::chrono::high_resolution_clock::now();
    mm_kernel(a.data(), b.data(), c.data(), size, size, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time taken by mm_kernel: " << elapsed.count() << " ms" << std::endl;

    // print the first 10 elements of the result matrix
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // compare to mm_naive
    std::vector<double> c_ref(size * size);
    mm_naive(a.data(), b.data(), c_ref.data(), size, size, size);
    bool equal = true;
    for (int i = 0; i < size * size; ++i) {
        if (abs(c[i] - c_ref[i]) > 1e-6) {
            equal = false;
            break;
        }
    }
    std::cout << "Result of mm_kernel is " << (equal ? "correct" : "incorrect") << std::endl;

    return 0;
}