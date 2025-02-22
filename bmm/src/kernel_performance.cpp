#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include "aligned_allocator.h"
#include "kernels.h"
#include "bmm.h"

using namespace std;
using namespace std::chrono;

template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

using namespace std;

void generate_random_matrix(aligned_vector<double> &matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

void measure_kernel_performance(const char* kernel_name,
                                void (*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int),
                                int num_repeats, int h, int w, ofstream &csv_file) {
    const int bd = 1;
    const int a_rows = 1024;
    const int b_cols = 1024;
    const int a_cols = 1024;
    const int simd_length = 4;
    const int wl = w / simd_length;
    const int b1 = 32;
    const int b2_ = 64;
    const int b3_ = 128;

    aligned_vector<double> a(bd * a_rows * a_cols);
    aligned_vector<double> b(bd * a_cols * b_cols);
    aligned_vector<double> c(bd * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(bd * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    bmm_naive(a.data(), b.data(), c_ref.data(), bd, a_rows, b_cols, a_cols);

    double total_time = 0.0;
    bool correct = true;

    for (int i = 0; i < num_repeats; ++i) {
        fill(c.begin(), c.end(), 0.0);

        auto start = high_resolution_clock::now();
        bmm(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols, h, w, simd_length, wl, b1, b2_, b3_, kernel);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;
        total_time += elapsed.count();

        for (int j = 0; j < bd * a_rows * b_cols; ++j) {
            if (abs(c[j] - c_ref[j]) > 1e-6) {
                correct = false;
                break;
            }
        }
    }

    double avg_time = total_time / num_repeats;
    csv_file << kernel_name << "," << avg_time << "," << (correct ? "correct" : "incorrect") << endl;
}

int main() {
    ofstream csv_file("kernel_performance.csv");
    csv_file << "Kernel,Time,Correctness" << endl;

    const int num_repeats = 1;

    cout << "Measuring kernel performance..." << endl;
    cout << "2x24" << endl;
    measure_kernel_performance("kernel_2x24", kernel_2x24, num_repeats, 2, 24, csv_file);
    cout << "4x4" << endl;
    measure_kernel_performance("kernel_4x4", kernel_4x4, num_repeats, 4, 4, csv_file);
    cout << "4x8" << endl;
    measure_kernel_performance("kernel_4x8", kernel_4x8, num_repeats, 4, 8, csv_file);
    cout << "4x12" << endl;
    measure_kernel_performance("kernel_4x12", kernel_4x12, num_repeats, 4, 12, csv_file);
    cout << "4x16" << endl;
    measure_kernel_performance("kernel_4x16", kernel_4x16, num_repeats, 4, 16, csv_file);
    cout << "4x20" << endl;
    measure_kernel_performance("kernel_4x20", kernel_4x20, num_repeats, 4, 20, csv_file);
    cout << "6x4" << endl;
    measure_kernel_performance("kernel_6x4", kernel_6x4, num_repeats, 6, 4, csv_file);
    cout << "6x8" << endl;
    measure_kernel_performance("kernel_6x8", kernel_6x8, num_repeats, 6, 8, csv_file);
    cout << "6x12" << endl;
    measure_kernel_performance("kernel_6x12", kernel_6x12, num_repeats, 6, 12, csv_file);
    cout << "6x16" << endl;
    measure_kernel_performance("kernel_6x16", kernel_6x16, num_repeats, 6, 16, csv_file);
    cout << "6x20" << endl;
    measure_kernel_performance("kernel_6x20", kernel_6x20, num_repeats, 6, 20, csv_file);
    cout << "8x4" << endl;
    measure_kernel_performance("kernel_8x4", kernel_8x4, num_repeats, 8, 4, csv_file);
    cout << "8x8" << endl;
    measure_kernel_performance("kernel_8x8", kernel_8x8, num_repeats, 8, 8, csv_file);
    cout << "8x12" << endl;
    measure_kernel_performance("kernel_8x12", kernel_8x12, num_repeats, 8, 12, csv_file);
    cout << "8x16" << endl;
    measure_kernel_performance("kernel_8x16", kernel_8x16, num_repeats, 8, 16, csv_file);
    cout << "8x20" << endl;
    measure_kernel_performance("kernel_8x20", kernel_8x20, num_repeats, 8, 20, csv_file);

    csv_file.close();

    cout << "Kernel performance results written to kernel_performance.csv" << endl;

    return 0;
}