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
#include <fstream>


// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;
//using aligned_vector_simd = std::vector<__m256d, aligned_allocator<__m256d, 64>>;

//using namespace std;

//const int h = 6; // kernel height
//const int w = 8; // kernel width
//const int simd_length = 4; // number of doubles in a SIMD register
//const int wl = 2; //width divided by simd_length
//// TODO: In the end hardcode height and width

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



void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl) {
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
            kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows_padded, b_cols_padded, a_cols, i, j, 0, a_cols, h, w, simd_length, wl);
//	            for (int k = 0; k < a_cols; k += 32) {
//                      kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows_padded, b_cols_padded, a_cols, i, j, k, std::min(k + 32, a_cols), h, w, simd_length, wl);
//                      }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int i = 0; i < a_rows; ++i) {
        std::memcpy(&c[i * b_cols], &c_aligned[i * b_cols_padded], b_cols * sizeof(double));
    }
}

void mm_blocked_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols, int h, int w, int simd_length, int wl, int b1, int b2_, int b3_) {
    // Pad a_rows and b_cols to be multiples of h and w
    int a_rows_padded = a_rows + (h - a_rows % h) % h;
    int b_cols_padded = b_cols + (w - b_cols % w) % w;

    // Block sizes need to be multiples of h and w
//    int b1 = b1_ - (b1_ % simd_length);
    int b2 = b2_ - (b2_ % h);
    int b3 = b3_ - (b3_ % w);

    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned;
    aligned_vector<double> c_aligned;
    pack(a, b, a_aligned, b_aligned, c_aligned, a_rows, a_cols, b_cols, a_rows_padded, b_cols_padded);

    // Perform block matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i3 = 0; i3 < b_cols_padded; i3 += b3) {
        for (int i2 = 0; i2 < a_rows_padded; i2 += b2) {
            for (int i1 = 0; i1 < a_cols; i1 += b1) {
                for (int k = i2; k < std::min(i2 + b2, a_rows_padded); k += h) {
                    for (int j = i3; j < std::min(i3 + b3, b_cols_padded); j += w) {
                        kernel(a_aligned.data(), b_aligned.data(), c_aligned.data(), a_rows_padded, b_cols_padded, a_cols, k, j, i1, std::min(i1 + b1, a_cols) , h, w, simd_length, wl);
                    }
                }//i1, std::min(i1 + b1, a_cols)
            }
        }
    }

    // Copy data from aligned memory to original matrix C, removing padding
    for (int i = 0; i < a_rows; ++i) {
        std::memcpy(&c[i * b_cols], &c_aligned[i * b_cols_padded], b_cols * sizeof(double));
    }
}


void kernel(double *a_aligned, double *b_aligned, double *c_aligned, const int a_rows, const int b_cols, const int a_cols,
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
        	for (int k = 0; k < simd_length; ++k) {
            	c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length + k] += t[i * wl + j][k];
            }
        }
    }


//    // Update c with the values in t, c += t
//	for (int i = 0; i < h; ++i) {
//    	for (int j = 0; j < wl; ++j) {
//    	    __m256d c_val = _mm256_load_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length]);
//    	    __m256d t_val = t[i * wl + j];
//    	    __m256d result = _mm256_add_pd(c_val, t_val);
//    	    _mm256_store_pd(&c_aligned[(a_idx + i) * b_cols + b_idx + j * simd_length], result);
//    	}
//	}
}


int main() {
  // mulktiply two random matrices of size 1024 x 1024 using mm_kernel
    int size = 512;
    std::vector<double> a(size * size), b(size * size), c(size * size);
    for (int i = 0; i < size * size; ++i) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // measure the time taken by mm_kernel
    auto start = std::chrono::high_resolution_clock::now();
    mm_kernel(a.data(), b.data(), c.data(), size, size, size, 8, 8, 4, 2);
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


    // Test correctness of mm_blocked_kernel
    int b1 = 32, b2 = 64, b3 = 128;
    int h = 3, w = 12;
    int simd_length = 4;
    int wl = w / simd_length;

    std::vector<double> c_blocked(size * size);
    start = std::chrono::high_resolution_clock::now();
    mm_blocked_kernel(a.data(), b.data(), c_blocked.data(), size, size, size, h, w, simd_length, wl, b1, b2, b3);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken by mm_blocked_kernel: " << elapsed.count() << " ms" << std::endl;

    // print the first 10 elements of the result matrix
    for (int i = 0; i < 10; ++i) {
        std::cout << c_blocked[i] << " ";
    }
    std::cout << std::endl;

    equal = true;
    for (int i = 0; i < size * size; ++i) {
        if (abs(c_ref[i] - c_blocked[i]) > 1e-6) {
            equal = false;
            break;
        }
    }
    std::cout << "Result of mm_blocked_kernel is " << (equal ? "correct" : "incorrect") << std::endl;


//    // For different height, width and matrix sizes time mm_kernel
////    int simd_length = 4;
//    std::string output_file = "kernel_size_results.csv";
//
//    std::vector<int> matrix_sizes = {703, 1024, 1500, 2048};
//    std::vector<int> heights = {4, 6, 8, 12};
//
//    std::vector<int> widths = {4, 8, 12, 16, 20};
//
//    int num_repeats = 2;
//
//    std::ofstream ofs(output_file);
//    if (!ofs.is_open()) {
//        std::cerr << "Failed to open output file: " << output_file << std::endl;
//        return -1;
//    }
//
//    ofs << "Matrix Size,Height,Width,Average Time (ms)" << std::endl;
//
//    std::random_device rd;
//    std::mt19937 gen(rd());
//
//    for (int size : matrix_sizes) {
//        std::uniform_real_distribution<> dis(0.0, 1.0);
//        std::vector<double> a(size * size), b(size * size), c(size * size);
//
//        for (auto &val : a) val = dis(gen);
//        for (auto &val : b) val = dis(gen);
//
//        for (int height : heights) {
//            for (int width : widths) {
//            	int wl = width / simd_length;
//
//                double total_time = 0.0;
//
//                for (int repeat = 0; repeat < num_repeats; ++repeat) {
//                    std::fill(c.begin(), c.end(), 0.0);
//
//                    auto start = std::chrono::high_resolution_clock::now();
//                    mm_kernel(a.data(), b.data(), c.data(), size, size, size, height, width, simd_length, wl);
//                    auto end = std::chrono::high_resolution_clock::now();
//
//                    std::chrono::duration<double, std::milli> elapsed = end - start;
//                    total_time += elapsed.count();
//                }
//
//                double average_time = total_time / num_repeats;
//                ofs << size << "," << height << "," << width << "," << average_time << std::endl;
//            }
//        }
//    }
//
//    ofs.close();
//
//    std::cout << "Benchmark results saved to " << output_file << std::endl;

    return 0;
}