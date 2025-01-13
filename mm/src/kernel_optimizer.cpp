
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <fstream>
#include <random>



using namespace std;

void kernel(double *aligned_a, double *aligned_b, double *c, const int a_rows, const int b_cols, const int a_cols,
               int a_idx, int b_idx, int height, int width, int l, int r) {
    // Assumption: A and B already aligned and B transposed
    // C gets updated - Initialization happens before.
    // Update C (add to c) for cols l to r from a and rows l to r from b (In the end from l=0 to r=a_cols must be called)

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = a_idx; i < a_idx + height; ++i) {
        for (int j = b_idx; j < b_idx + width; ++j) {
            __m256d c_vec1 = _mm256_setzero_pd();
            __m256d c_vec2 = _mm256_setzero_pd();
            __m256d c_vec3 = _mm256_setzero_pd();
            __m256d c_vec4 = _mm256_setzero_pd();
            __m256d c_vec5 = _mm256_setzero_pd();
            __m256d c_vec6 = _mm256_setzero_pd();
            __m256d c_vec7 = _mm256_setzero_pd();
            __m256d c_vec8 = _mm256_setzero_pd();
            for (int k = l; k < r; k += 32) {
                __m256d a_vec1 = _mm256_load_pd(&aligned_a[i * a_cols + k]);
                __m256d b_vec1 = _mm256_load_pd(&aligned_b[j * a_cols + k]);
                c_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, c_vec1);

                __m256d a_vec2 = _mm256_load_pd(&aligned_a[i * a_cols + k + 4]);
                __m256d b_vec2 = _mm256_load_pd(&aligned_b[j * a_cols + k + 4]);
                c_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, c_vec2);

                __m256d a_vec3 = _mm256_load_pd(&aligned_a[i * a_cols + k + 8]);
                __m256d b_vec3 = _mm256_load_pd(&aligned_b[j * a_cols + k + 8]);
                c_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, c_vec3);

                __m256d a_vec4 = _mm256_load_pd(&aligned_a[i * a_cols + k + 12]);
                __m256d b_vec4 = _mm256_load_pd(&aligned_b[j * a_cols + k + 12]);
                c_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, c_vec4);

                __m256d a_vec5 = _mm256_load_pd(&aligned_a[i * a_cols + k + 16]);
                __m256d b_vec5 = _mm256_load_pd(&aligned_b[j * a_cols + k + 16]);
                c_vec5 = _mm256_fmadd_pd(a_vec5, b_vec5, c_vec5);

                __m256d a_vec6 = _mm256_load_pd(&aligned_a[i * a_cols + k + 20]);
                __m256d b_vec6 = _mm256_load_pd(&aligned_b[j * a_cols + k + 20]);
                c_vec6 = _mm256_fmadd_pd(a_vec6, b_vec6, c_vec6);

                __m256d a_vec7 = _mm256_load_pd(&aligned_a[i * a_cols + k + 24]);
                __m256d b_vec7 = _mm256_load_pd(&aligned_b[j * a_cols + k + 24]);
                c_vec7 = _mm256_fmadd_pd(a_vec7, b_vec7, c_vec7);

                __m256d a_vec8 = _mm256_load_pd(&aligned_a[i * a_cols + k + 28]);
                __m256d b_vec8 = _mm256_load_pd(&aligned_b[j * a_cols + k + 28]);
                c_vec8 = _mm256_fmadd_pd(a_vec8, b_vec8, c_vec8);
            }
            c[i * b_cols + j] += c_vec1[0] + c_vec1[1] + c_vec1[2] + c_vec1[3] +
                                 c_vec2[0] + c_vec2[1] + c_vec2[2] + c_vec2[3] +
                                 c_vec3[0] + c_vec3[1] + c_vec3[2] + c_vec3[3] +
                                 c_vec4[0] + c_vec4[1] + c_vec4[2] + c_vec4[3] +
                                 c_vec5[0] + c_vec5[1] + c_vec5[2] + c_vec5[3] +
                                 c_vec6[0] + c_vec6[1] + c_vec6[2] + c_vec6[3] +
                                 c_vec7[0] + c_vec7[1] + c_vec7[2] + c_vec7[3] +
                                 c_vec8[0] + c_vec8[1] + c_vec8[2] + c_vec8[3];
        }
    }
}

void mm_kernel(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols, int height, int width) {
    // Allocate aligned memory for matrices A and B
    double *aligned_a, *aligned_b;
    posix_memalign(reinterpret_cast<void**>(&aligned_a), 64, a_rows * a_cols * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&aligned_b), 64, a_cols * b_cols * sizeof(double));

    // Copy data from original matrix A to aligned memory
    std::memcpy(aligned_a, a, a_rows * a_cols * sizeof(double));

    // Transpose and copy data from original matrix B to aligned memory
    for (int i = 0; i < a_cols; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            aligned_b[i * b_cols + j] = b[j * a_cols + i];
        }
    }

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // TODO: Pad Matrices so that dimensions are multiples of 4 and shared dim is multiple of 32

    // Perform matrix multiplication using AVX intrinsics with pipelined FMA calls
    for (int i = 0; i < a_rows; i += height) { // TODO: use a_rows_padded here
        for (int j = 0; j < b_cols; j += width) { // TODO: use b_cols_padded here
//            std::cout << "i: " << i << " j: " << j << std::endl;
            kernel(aligned_a, aligned_b, c, a_rows, b_cols, a_cols, i, j, height, width, 0, a_cols);
            // for now l=0 and r=a_cols, later in block matrix multiplication, l and r will be updated
        }
    }

    // Clean up
    free(aligned_a);
    free(aligned_b);
}


void run_kernel_benchmarks(const std::string &output_file, const std::vector<int> &matrix_sizes, const int num_repeats,
                           const std::vector<int> &heights, const std::vector<int> &widths) {
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    ofs << "Matrix Size,Height,Width,Average Time (ms)" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int size : matrix_sizes) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<double> a(size * size), b(size * size), c(size * size);

        for (auto &val : a) val = dis(gen);
        for (auto &val : b) val = dis(gen);

        for (int height : heights) {
            for (int width : widths) {

                double total_time = 0.0;

                for (int repeat = 0; repeat < num_repeats; ++repeat) {
                    std::fill(c.begin(), c.end(), 0.0);

                    auto start = std::chrono::high_resolution_clock::now();
                    mm_kernel(a.data(), b.data(), c.data(), size, size, size, height, width);
                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> elapsed = end - start;
                    total_time += elapsed.count();
                }

                double average_time = total_time / num_repeats;
                ofs << size << "," << height << "," << width << "," << average_time << std::endl;
            }
        }
    }

    ofs.close();
}

int main() {
    std::string output_file = "kernel_optimizer_results.csv";
    std::vector<int> matrix_sizes = {1024};
    int num_repeats = 1;

    std::vector<int> heights = {2, 4, 8, 16, 32, 64, 128};
    std::vector<int> widths = {2, 4, 8, 16, 32, 64, 128};

    run_kernel_benchmarks(output_file, matrix_sizes, num_repeats, heights, widths);

    std::cout << "Benchmark results saved to " << output_file << std::endl;
    return 0;
}