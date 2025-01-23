#include "aligned_allocator.h"
#include "mm.h"
#include <vector>  // std::vector for using aligned_vector
#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>

using namespace std;

// aligned_vector is a 64 byte aligned std::vector
template <class T>
using aligned_vector = std::vector<T, alligned_allocator<T, 64>>;
//using aligned_vector = std::vector<T>;


void mm_blocked_packed_stdmin_wrapper(const double *a, const double *b, double *c, const int a_rows, const int b_cols, const int a_cols, int b3, int b2, int b1) {
    // Pack data
    aligned_vector<double> a_aligned;
    aligned_vector<double> b_aligned_transposed;
    pack(a, b, a_aligned, b_aligned_transposed, a_rows, a_cols, b_cols);

    // Initialize result matrix c to zero
    for (int i = 0; i < a_rows * b_cols; ++i)
        c[i] = 0.0;

    // kernel size
    int height = 4;
    int width = 4;

    // Block matrix multiplication
    for (int i3 = 0; i3 < b_cols; i3 += b3) {
        for (int i2 = 0; i2 < a_cols; i2 += b2) {
            for (int i1 = 0; i1 < a_rows; i1 += b1) {
                // multiply the block
                for (int i = i2; i < ((i2+b2) < a_cols ? i2+b2 : a_cols); i += height) {
                    for (int j = i3; j < ((i3 + b3) < b_cols ? i3 + b3 : b_cols); j += width) {
                        kernel(a_aligned.data(), b_aligned_transposed.data(), c, a_rows, b_cols, a_cols, i, j, height, width, i1, ((i1+b1) < a_rows ? i1+b1 : a_rows));
                    }
                }
            }
        }
    }
}

int main() {
    int num_repeats = 2;

    // Example matrix sizes
    int a_rows = 2048, b_cols = 2048, a_cols = 2048;
    std::vector<double> a(a_rows * a_cols);
    std::vector<double> b(a_cols * b_cols);
    std::vector<double> c(a_rows * b_cols, 0.0);

    // fill matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto &val : a) val = dis(gen);
    for (auto &val : b) val = dis(gen);

    // Block sizes to test
    std::vector<int> b3_sizes = {16, 32, 64, 128, 256};
    std::vector<int> b2_sizes = {16, 32, 64, 128, 256};
    std::vector<int> b1_sizes = {16, 32, 64, 128, 256};

    // Open CSV file for writing
    std::ofstream csv_file("block_optimizer_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open output file: " << "block_optimizer_results.csv" << std::endl;
        return -1;
    }
    csv_file << "b3,b2,b1,Average time (ms)\n";

    for (int b3 : b3_sizes) {
        for (int b2 : b2_sizes) {
            for (int b1 : b1_sizes) {

                if (b3 > b2 || b2 > b1) {
                    continue;
                }

                double total_time = 0.0;
                cout << "Running for b3: " << b3 << " b2: " << b2 << " b1: " << b1 << endl;

                for (int repeat = 0; repeat < num_repeats; ++repeat) {
                    std::fill(c.begin(), c.end(), 0.0);

                    auto start = std::chrono::high_resolution_clock::now();
                    mm_blocked_packed_stdmin_wrapper(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, b3, b2, b1);
                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> elapsed = end - start;
                    total_time += elapsed.count();
                }
            double average_time = total_time / num_repeats;
            csv_file << b3 << "," << b2 << "," << b1 << "," << average_time << std::endl;
            }
        }
    }

    cout << "Benchmark results saved to block_optimizer_results.csv" << endl;

    // Close CSV file
    csv_file.close();

    return 0;
}