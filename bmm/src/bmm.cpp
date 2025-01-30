#include "bmm.h"
#include "aligned_allocator.h"
#include <vector>  // std::vector for using aligned_vector
#include <immintrin.h>
#include <cstring> // std::memcpy
#include <iostream> // std::cerr
//#include <algorithm> // std::fill //todo: ??
#include <chrono> // std::chrono


// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;


// naive approach
double *batch_matrix_multiply(const double *a, const double *b, double *c,
    const int batch_dim, const int a_rows, const int b_cols, const int a_cols) {
    for (int d = 0; d < batch_dim; ++d)
        for (int i = 0; i < a_rows; ++i)
            for (int j = 0; j < b_cols; ++j)
                for (int k = 0; k < a_cols; ++k)
                    c[d*a_rows*b_cols + i*b_cols + j] += a[d*a_rows*a_cols + i*a_cols + k] * b[d*a_cols*b_cols + k*b_cols + j];
    return c;
}

// optimized approach
// 1. Preprocessing/packing function to align matrices A and B and C, (!) B is NOT transposed, c is initialized to zero
// 2. Matrix multiplication kernel (2 versions with AVX2 + version with OpenMP for general architectures)
// 3. Use Blocking to improve cache performance
// 4. Parallelize the computation using OpenMP
// 5. Copy data from aligned memory to original matrix C, removing padding
// for now: block sizes and kernel size as parameter
// Later: Hardcode block sizes and kernel size, UNROLL KERNEL LOOPS

void pack(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
          aligned_vector<double> &c_aligned, const int batch_dim, const int a_rows, const int b_cols, const int a_cols,
          const int a_rows_padded, const int b_cols_padded, const int a_cols_padded) {
    // Allocate aligned memory for matrices A, B, and C with padded dimensions
    a_aligned.resize(batch_dim * a_rows_padded * a_cols);
    b_aligned.resize(batch_dim * a_cols * b_cols_padded);
    c_aligned.resize(batch_dim * a_rows_padded * b_cols_padded);

    // Copy data from original matrix A to aligned memory and set padded elements to 0
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows_padded; ++i) {
            if (i < a_rows) {
                std::memcpy(&a_aligned[d * a_rows_padded * a_cols + i * a_cols], &a[d * a_rows * a_cols + i * a_cols], a_cols * sizeof(double));
            } else {
                std::fill(&a_aligned[d * a_rows_padded * a_cols + i * a_cols], &a_aligned[d * a_rows_padded * a_cols + (i + 1) * a_cols], 0.0);
            }
        }
    }

    // Copy data from original matrix B to aligned memory and set padded elements to 0
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_cols; ++i) {
            if (i < a_cols) {
                std::memcpy(&b_aligned[d * a_cols * b_cols_padded + i * b_cols_padded], &b[d * a_cols * b_cols + i * b_cols], b_cols * sizeof(double));
                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols_padded + b_cols], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols_padded], 0.0);
            } else {
                std::fill(&b_aligned[d * a_cols * b_cols_padded + i * b_cols_padded], &b_aligned[d * a_cols * b_cols_padded + (i + 1) * b_cols_padded], 0.0);
            }
        }
    }

    // Initialize result matrix c to zero
    std::fill(c_aligned.begin(), c_aligned.end(), 0.0);
}

void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int bd, const int a_rows, const int b_cols, const int a_cols,
             const int h, const int wl, int a_idx, int b_idx, int l, int r) {

    // Create a vector of __m256d with size height * width
    aligned_vector<__m256d> t(h * wl); // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
    // t[i]: i-th element of t (a 4 element _m256d SIMD vector), t[i][j]: j-th double of i-th element of t

    // Initialize each element with zeros
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < h * wl; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        for (int j = 0; j < wl; j++) {
            __m256d b0 = _mm256_load_pd(&b_aligned[bd * a_cols * b_cols + k * b_cols + b_idx + j * 4]); // simd_length == 4
            for (int i = 0; i < h; i++) {
                __m256d a0 = _mm256_broadcast_sd(&a_aligned[bd * a_rows * a_cols + (a_idx + i) * a_cols + k]);
                t[i * wl + j] = _mm256_fmadd_pd(a0, b0, t[i * wl + j]);
            }
        }
    }

    // Update c with the values in t
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < wl; ++j) {
            _mm256_store_pd(&c_aligned[bd * a_rows * b_cols + (a_idx + i) * b_cols + b_idx + j * 4], t[i * wl + j]); // simd_length == 4
        }
    }
}


void bmm(const double *a, const double *b, double *c, const int batch_dim, const int a_rows, const int b_cols, const int a_cols,
         const int h, const int w, const int b1, const int b2, const int b3) {

    // Check that b2 is multiple of h and b3 is multiple of w
    if (b2 % h != 0 || b3 % w != 0) {
        std::cerr << "Error: b2 must be a multiple of h and b3 must be a multiple of w" << std::endl;
        return;
    }

    // define wl == width divided by simd_length
    const int wl = w / 4; // 4 doubles in a SIMD register

    // Pad b_cols to be a multiple of b3, a_cols to be a multiple of b2, and a_rows to be a multiple of b1
    int a_rows_padded = a_rows + (b1 - a_rows % b1) % b1;
    int b_cols_padded = b_cols + (b3 - b_cols % b3) % b3;
    int a_cols_padded = a_cols + (b2 - a_cols % b2) % b2;

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, batch_dim, a_rows, b_cols, a_cols, a_rows_padded, b_cols_padded, a_cols_padded);

    // Blocked MM with kernel size h x w
    for (int d = 0; d < batch_dim; ++d) {
        for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
            for (int i2 = 0; i2 < a_cols_padded; i2 += b2) {
                for (int i1 = 0; i1 < a_rows_padded; i1 += b1) {
                    // multiply the block using the kernel
                    for (int i = i2; i < a_cols_padded; i += h) { // std::min not requiered because of padding, std::min(i2+b2, a_cols) == a_cols
                        for (int j = i3; j < b_cols_padded; j += w) { // std::min not requiered because of padding, std::min(i3 + b3, b_cols) == b_cols
                            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), d, a_rows_padded, b_cols_padded, a_cols_padded, h, wl, i, j, i1, a_rows); // std::min not requiered because of padding, std::min(i1+b1, a_rows) == a_rows
                        }
                    }
                }
            }
        }
    }


    // Copy data from aligned memory to original matrix C, removing padding
    for (int d = 0; d < batch_dim; ++d) {
        for (int i = 0; i < a_rows; ++i) {
            std::memcpy(&c[d * a_rows * b_cols + i * b_cols], &c_aligned[d * a_rows_padded * b_cols_padded + i * b_cols_padded], b_cols * sizeof(double));
        }
    }
}

// TODO: Prallelize over BD, other kernels, resgtricted pointers


//DEBUGGING

void kernel_mm(double *a_aligned, double *b_aligned, double *c_aligned, const int a_rows, const int b_cols, const int a_cols,
             int a_idx, int b_idx, int l, int r, int h, int w, int simd_length, int wl) {

    // Create a vector of __m256d with size height * width
    aligned_vector<__m256d> t(h * wl); // kernel size = height * width, temporary t stores 4 doubles in each element, -> h * wl = 6 * 2 = 12
    // t[i]: i-th element of t (a 4 element _m256d SIMD vector), t[i][j]: j-th double of i-th element of t

    // Initialize each element with zeros
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < h * wl; ++i) {
        t[i] = zero;
    }

    for (int k = l; k < r; k++) {
        for (int j = 0; j < wl; j++) {
            __m256d b0 = _mm256_load_pd(&b_aligned[k * b_cols + b_idx + j * simd_length]);
            for (int i = 0; i < h; i++) {
                __m256d a0 = _mm256_broadcast_sd(&a_aligned[(a_idx + i) * a_cols + k]);
                t[i * wl + j] = _mm256_fmadd_pd(a0, b0, t[i * wl + j]);
            }
        }
    }

    // Update c with the values in t
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < wl; ++j) {
            _mm256_store_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length], t[i * wl + j]);
        }
    }
}

void pack_mm(const double *a, const double *b, aligned_vector<double> &a_aligned, aligned_vector<double> &b_aligned,
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

void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack_mm(a, b, a_aligned, b_aligned, c_aligned, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows_padded; i += h) {
        for (int j = 0; j < b_cols_padded; j += w) {
            kernel_mm(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows_padded, b_cols_padded, a_cols, i, j, 0, a_cols, h, w, simd_length, wl);
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int i = 0; i < a_rows; ++i) {
        std::memcpy(&c[i * b_cols], &c_aligned[i * b_cols_padded], b_cols * sizeof(double));
    }
}

void mm_naive(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols) {
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];
}




int main() {
    // multiply matrices
    int size = 512;
    std::vector<double> a(size * size), b(size * size), c(size * size);
    for (int i = 0; i < size * size; ++i) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // measure the time taken by mm_kernel
    auto start = std::chrono::high_resolution_clock::now();
    mm_kernel(a.data(), b.data(), c.data(), size, size, size, 8, 16, 4, 2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken by mm_kernel: " << elapsed.count() << " seconds" << std::endl;

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