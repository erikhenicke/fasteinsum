#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <iomanip>
#include <unordered_map>

#include "aligned_allocator.h"
#include "kernels.h"
#include "bmm.h"

using namespace std;
using namespace std::chrono;

template <class T>
using aligned_vector = vector<T, aligned_allocator<T, 64>>;

using namespace std;

constexpr int SIMD_LENGTH = 4;

void delete_last_printed(const string& str) {
    for (size_t i = 0; i < str.length(); ++i) {
        cout << "\b \b";
    }
}

void generate_random_matrix(aligned_vector<double> &matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

void flush_cache() {
    const size_t cache_size = 256 * 1024 * 1024; // 256 MB
    vector<char> dummy(cache_size, 1);
    #pragma omp parallel for
    for (size_t i = 0; i < dummy.size(); ++i) {
        dummy[i] += 1;
    }
}

bool check_correctness(
    int h,
    int w,
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2_,
    int b3_,
    void (*bmm)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, int, int, int, int,
         void(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)),
    void(*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int))
{
    const int wl = w / SIMD_LENGTH;

    aligned_vector<double> a(batch_dim * a_rows * a_cols);
    aligned_vector<double> b(batch_dim * a_cols * b_cols);
    aligned_vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    // std::fill(c.begin(), c.end(), 0.0);
    // std::fill(c_ref.begin(), c_ref.end(), 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    bool correct = true;

    // Correctness check
    bmm_naive(a.data(), b.data(), c_ref.data(), batch_dim, a_rows, b_cols, a_cols);
    bmm(a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, h, w, SIMD_LENGTH, wl, b1, b2_, b3_, kernel);

    for (int j = 0; j < batch_dim * a_rows * b_cols; ++j) {
        if (abs(c[j] - c_ref[j]) > 1e-6) {
            correct = false;
            break;
        }
    }

    return correct;
}

double measure_kernel_performance(
    int num_repeats,
    int h,
    int w,
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2_,
    int b3_,
    void (*bmm)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, int, int, int, int,
         void(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)),
    void(*kernel)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int))
{
    const int wl = w / SIMD_LENGTH;

    aligned_vector<double> a(batch_dim * a_rows * a_cols);
    aligned_vector<double> b(batch_dim * a_cols * b_cols);
    aligned_vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    vector<double> times;

    for (int i = 0; i < num_repeats; ++i) {
        fill(c.begin(), c.end(), 0.0);

        // Flush the cache
        flush_cache();

        auto start = high_resolution_clock::now();
        bmm(a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, h, w, SIMD_LENGTH, wl, b1, b2_, b3_, kernel);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    // Calculate median time
    sort(times.begin(), times.end());
    double median_time = times[times.size() / 2];

    return median_time;
}

// void measure_naive_performance(int num_repeats, ofstream &csv_file) {
//     const int bd = 1;
//     const int a_rows = 2048;
//     const int b_cols = 2048;
//     const int a_cols = 2048;
//
//     aligned_vector<double> a(bd * a_rows * a_cols);
//     aligned_vector<double> b(bd * a_cols * b_cols);
//     aligned_vector<double> c(bd * a_rows * b_cols, 0.0);
//
//     generate_random_matrix(a, a_rows, a_cols);
//     generate_random_matrix(b, a_cols, b_cols);
//
//     double total_time = 0.0;
//
//     for (int i = 0; i < num_repeats; ++i) {
//         fill(c.begin(), c.end(), 0.0);
//
//         auto start = high_resolution_clock::now();
//         bmm_naive(a.data(), b.data(), c.data(), bd, a_rows, b_cols, a_cols);
//         auto end = high_resolution_clock::now();
//
//         duration<double> elapsed = end - start;
//         total_time += elapsed.count();
//
//     }
//
//     double avg_time = total_time / num_repeats;
//     csv_file << "naive" << "," << avg_time << endl;
// }

int main() {
    const string fileName = "kernel_performance.csv";
    ofstream csv_file(fileName);
    csv_file << "Kernel,Time,Correctness" << endl;

    unordered_map<string, int> hashTable;

    const int num_repeats = 10;
    const int num_repeats_shuffle = 20;

    cout << "Measuring kernel performance..." << endl;

    vector<
        tuple<
            string,
            void (*)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, int, int, int, int,
             void(double*, double*, double*, const int, const int, const int, const int, int, int, int, int)),
            void(*)(double*, double*, double*, const int, const int, const int, const int, int, int, int, int),
            int,
            int,
            double,
            bool>>
    kernels = {
//         {"kernel_2x24", bmm_parallel, kernel_2x24, 2, 24, 0.0, false},
//         {"kernel_4x4", bmm_parallel, kernel_4x4, 4, 4, 0.0, false},
//         {"kernel_4x8", bmm_parallel, kernel_4x8, 4, 8, 0.0, false},
//         {"kernel_4x12", bmm_parallel, kernel_4x12, 4, 12, 0.0, false},
//         {"kernel_4x12_test1", bmm_parallel, kernel_4x12_test1, 4, 12, 0.0, false},
//         {"kernel_4x12_test2", bmm_parallel, kernel_4x12_test2, 4, 12, 0.0, false},
//         {"kernel_4x16", bmm_parallel, kernel_4x16, 4, 16, 0.0, false},
//         {"kernel_4x20", bmm_parallel, kernel_4x20, 4, 20, 0.0, false},
//         {"kernel_6x4", bmm_parallel, kernel_6x4, 6, 4, 0.0, false},
//         {"kernel_6x8", bmm_parallel, kernel_6x8, 6, 8, 0.0, false},
//         {"kernel_6x12", bmm_parallel, kernel_6x12, 6, 12, 0.0, false},
//         {"kernel_6x16", bmm_parallel, kernel_6x16, 6, 16, 0.0, false},
//         {"kernel_6x20", bmm_parallel, kernel_6x20, 6, 20, 0.0, false},
//         {"kernel_8x4", bmm_parallel, kernel_8x4, 8, 4, 0.0, false},
//         {"kernel_8x8", bmm_parallel, kernel_8x8, 8, 8, 0.0, false},
//         {"kernel_8x12", bmm_parallel, kernel_8x12, 8, 12, 0.0, false},
        {"kernel_8x16 parallel", bmm_parallel, kernel_8x16, 8, 16, 0.0, false},
//         {"kernel_8x16 parallel 1", bmm_parallel_more1, kernel_8x16, 8, 16, 0.0, false},
//         {"kernel_8x16 parallel 2", bmm_parallel_more2, kernel_8x16, 8, 16, 0.0, false},
//         {"kernel_8x16 parallel 3", bmm_parallel_more3, kernel_8x16, 8, 16, 0.0, false},
        {"kernel_8x16 parallel 4", bmm_parallel_more4, kernel_8x16, 8, 16, 0.0, false},
        {"kernel_8x16 parallel 5", bmm_parallel_more5, kernel_8x16, 8, 16, 0.0, false},};
//        {"kernel_8x16_test2", bmm_parallel, kernel_8x16_test2, 8, 16, 0.0, false},
//         {"kernel_8x20", bmm_parallel, kernel_8x20, 8, 20, 0.0, false}};

    vector<tuple<const int, const int, const int, const int, const int, const int, const int>> sizes = {
        {16, 2048, 512, 2048, 128, 512, 1024}};

    const size_t num_configs = sizes.size() * kernels.size();

    // Create a vector of kernel names to preserve the output order
    vector<string> kernel_names;
    for (const auto& kernel : kernels) {
        const string name = get<0>(kernel);
        kernel_names.push_back(name);
    }

    cout << "Check correctness: " << endl;
    for (const auto& size : sizes) {
        for (auto& kernel : kernels) {
            bool isCorrect = check_correctness(
                get<3>(kernel),
                get<4>(kernel),
                get<0>(size),
                get<1>(size),
                get<2>(size),
                get<3>(size),
                get<4>(size),
                get<5>(size),
                get<6>(size),
                get<1>(kernel),
                get<2>(kernel));
            get<6>(kernel) = isCorrect;
            cout << "\tKernel: " << get<0>(kernel) << " " << (isCorrect ? "correct" : "incorrect") << endl;
        }
    }

    random_device rd;
    mt19937 g(rd());

    for (int i = 0; i < num_repeats_shuffle; ++i) {
        cout << "Iteration: " << setw(3) << i + 1 << "/" << num_repeats_shuffle << endl;

        int curr_it = 0;

        // Randomize the order of execution
        shuffle(kernels.begin(), kernels.end(), g);

        for (const auto& size : sizes) {
            for (auto& kernel : kernels) {
                cout << "\tKernel: " << get<0>(kernel) << " (" << curr_it + 1 << "/" << num_configs << ")" << endl;

                get<5>(kernel) += measure_kernel_performance(
                    num_repeats,
                    get<3>(kernel),
                    get<4>(kernel),
                    get<0>(size),
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    get<1>(kernel),
                    get<2>(kernel));

                curr_it++;
            }
        }
    }

    // Output the results in the original order
    for (const string name : kernel_names) {
        for (const auto& kernel: kernels)
            if (name == get<0>(kernel)) {
                csv_file << get<0>(kernel) << ","
                << (get<5>(kernel) / num_repeats_shuffle) << ","
                << (get<6>(kernel) ? "correct" : "incorrect") << endl;
            }
    }

    csv_file.close();

    cout << "Kernel performance results written to " << fileName << endl;

    return 0;
}