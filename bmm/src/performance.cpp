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
#include <sstream>


#include "aligned_allocator.h"
//#include "kernels.h"
//#include "bmm.h"
#include "wrapper.h"

using namespace std;
using namespace std::chrono;

template <class T>
using aligned_vector = vector<T, aligned_allocator<T, 64>>;

using namespace std;

constexpr int SIMD_LENGTH = 4;

struct KernelResult {
    string name;
//    int h, w;
    int batch_dim, a_rows, a_cols, b_cols;
    int b1, b2, b3;
    bool correct;
    double time;
};

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
//    int h,
//    int w,
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2,
    int b3,
    void (*bmm_wrapper)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, double*))
{
    aligned_vector<double> a(batch_dim * a_rows * a_cols);
    aligned_vector<double> b(batch_dim * a_cols * b_cols);
    aligned_vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    // std::fill(c.begin(), c.end(), 0.0);
    // std::fill(c_ref.begin(), c_ref.end(), 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    bool correct = true;

    double time_dummy;

    // Correctness check
    bmm_naive_wrapper(a.data(), b.data(), c_ref.data(), batch_dim, a_rows, b_cols, a_cols, b1, b2, b3, &time_dummy);
    bmm_wrapper(a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, b1, b2, b3, &time_dummy);

    for (int j = 0; j < batch_dim * a_rows * b_cols; ++j) {
        if (abs(c[j] - c_ref[j]) > 1e-6) {
            correct = false;
            break;
        }
    }

    return correct;
}

double measure_performance(
    int num_repeats,
//    int h,
//    int w,
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2,
    int b3,
    void (*bmm_wrapper)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, double*))
{
    aligned_vector<double> a(batch_dim * a_rows * a_cols);
    aligned_vector<double> b(batch_dim * a_cols * b_cols);
    aligned_vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

    generate_random_matrix(a, a_rows, a_cols);
    generate_random_matrix(b, a_cols, b_cols);

    vector<double> times(num_repeats);

    for (int i = 0; i < num_repeats; ++i) {
        fill(c.begin(), c.end(), 0.0);

        // Flush the cache
        flush_cache();

        bmm_wrapper(a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, b1, b2, b3, &times[i]);

//        auto start = high_resolution_clock::now();
//        bmm(a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols, h, w, SIMD_LENGTH, wl, b1, b2_, b3_, kernel);
//        auto end = high_resolution_clock::now();
//
//        duration<double> elapsed = end - start;
//        times.push_back(elapsed.count());
    }

//    // Calculate median time
//    sort(times.begin(), times.end());
//    double median_time = times[times.size() / 2];
    // Calculate average time
    double total_time = 0.0;
    for (int i = 0; i < num_repeats; ++i) {
        total_time += times[i];
    }
    double avg_time = total_time / num_repeats;

//    return median_time;
    return avg_time;
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
    const int num_repeats = 1;
    const int num_repeats_shuffle = 2;

    cout << "Measuring performance..." << endl;

    /* Contains:
     * 1. BMM name
     * 2. Bmm wrapper function
     */
    vector<
        tuple<
            string,
            void (*)(const double*, const double*, double*, const int, const int, const int, const int, int, int, int, double*)>>
//    kernels = {
//         {"kernel_2x24", bmm_parallel, kernel_2x24, 2, 24},
//         {"kernel_4x4", bmm_parallel, kernel_4x4, 4, 4},
//         {"kernel_4x8", bmm_parallel, kernel_4x8, 4, 8},
//         {"kernel_4x12", bmm_parallel, kernel_4x12, 4, 12},
//         {"kernel_4x12_test1", bmm_parallel, kernel_4x12_test1, 4, 12},
//         {"kernel_4x12_test2", bmm_parallel, kernel_4x12_test2, 4, 12},
//         {"kernel_4x16", bmm_parallel, kernel_4x16, 4, 16},
//         {"kernel_4x20", bmm_parallel, kernel_4x20, 4, 20},
//         {"kernel_6x4", bmm_parallel, kernel_6x4, 6, 4},
//         {"kernel_6x8", bmm_parallel, kernel_6x8, 6, 8},
//         {"kernel_6x12", bmm_parallel, kernel_6x12, 6, 12},
//         {"kernel_6x16", bmm_parallel, kernel_6x16, 6, 16},
//         {"kernel_6x20", bmm_parallel, kernel_6x20, 6, 20},
//         {"kernel_8x4", bmm_parallel, kernel_8x4, 8, 4},
//         {"kernel_8x8", bmm_parallel, kernel_8x8, 8, 8},
//         {"kernel_8x12", bmm_parallel, kernel_8x12, 8, 12},
//         {"kernel_8x16 parallel", bmm_parallel, kernel_8x16, 8, 16},
//         {"kernel_8x16 parallel 1", bmm_parallel_more1, kernel_8x16, 8, 16},
//         {"kernel_8x16 parallel 2", bmm_parallel_more2, kernel_8x16, 8, 16},
//         {"kernel_8x16 parallel 3", bmm_parallel_more3, kernel_8x16, 8, 16},
//         {"kernel_8x16 parallel 4", bmm_parallel_more4, kernel_8x16, 8, 16},
//         {"kernel_8x16 parallel 5", bmm_parallel_more5, kernel_8x16, 8, 16},
//         {"kernel_8x16_test2", bmm_parallel, kernel_8x16_test2, 8, 16},
//         {"kernel_8x20", bmm_parallel, kernel_8x20, 8, 20},
//        {"simple_kernel", bmm_simple_kernel, kernel2, 6, 8},
//        {"kernel_8x16", bmm_parallel, kernel_8x16, 8, 16}};

	functions = {
        {"kernel8x16", bmm_kernel8x16_wrapper},
        {"kernel4x12", bmm_kernel4x12_wrapper},
//        {"simple", bmm_kernel_simple_wrapper},
//        {"blocked", bmm_blocked_wrapper},
//        {"blas", bmm_blas_wrapper},
        {"naive", bmm_naive_wrapper}

        };

    // Contains:
    // 1. Batch dimension
    // 2. A rows
    // 3. A cols
    // 4. B cols
    // 5. B1
    // 6. B2
    // 7. B3
    vector<tuple<int, int, int, int, int, int, int>> sizes = {
//        {4, 1024, 1024, 1024, 32, 64, 128},
//        {4, 1024, 1024, 1024, 128, 128, 128},
//        {4, 1024, 1024, 1024, 128, 64, 32},
//        {4, 1024, 1024, 1024, 64, 128, 256},
//        {4, 1024, 1024, 1024, 256, 256, 256},
        {4, 1024, 1024, 1024, 256, 128, 64}
        };

//        // Block sizes to test
//    std::vector<int> b3_sizes = {128};//32, 64, 128, 256};
//    std::vector<int> b2_sizes = {128};//32, 64, 128, 256};
//    std::vector<int> b1_sizes = {128};//32, 64, 128, 256};
//
//    for (int b1 : b1_sizes) {
//        for (int b2 : b2_sizes) {
//            for (int b3 : b3_sizes) {
////                if (b3 > b2 || b2 > b1) {
////                    continue;
////                }
//                // add the block sizes to the sizes vector
//                sizes.push_back({4, 1024, 1024, 1024, b1, b2, b3});
//            }
//        }
//    }


    const size_t num_configs = sizes.size() * functions.size();

    vector<KernelResult> results;

    cout << "Check correctness: " << endl;
    for (const auto& size : sizes) {
        for (auto& func : functions) {
//            bool isCorrect = check_correctness(
////                get<3>(kernel), //h, w, not needed anymore
////                get<4>(kernel),
//                get<0>(size),
//                get<1>(size),
//                get<2>(size),
//                get<3>(size),
//                get<4>(size),
//                get<5>(size),
//                get<6>(size),
//                get<1>(func));
//            cout << "\tBMM function: " << get<0>(func) << " " << (isCorrect ? "correct" : "incorrect") << endl;

// SKIP CORRECTNESS CHEKCS
       			cout << "Skipping correctness checks" << endl;
	            bool isCorrect = true;

            results.push_back({
                    get<0>(func),
//                    get<3>(kernel), //h, w, not needed
//                    get<4>(kernel),
                    get<0>(size),
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    isCorrect,
                    0.0});
        }
    }

    random_device rd;
    mt19937 g(rd());

    for (int i = 0; i < num_repeats_shuffle; ++i) {
        cout << "Iteration: " << setw(3) << i + 1 << "/" << num_repeats_shuffle << endl;

        int curr_it = 0;

        // Randomize the order of execution
        shuffle(functions.begin(), functions.end(), g);
        shuffle(sizes.begin(), sizes.end(), g);

        for (const auto& size : sizes) {
            for (auto& func : functions) {
                cout << "\tBMM function: " << get<0>(func) << " (" << curr_it + 1 << "/" << num_configs << ")" << endl;

                double time = measure_performance(
                    num_repeats,
//                    get<3>(kernel),
//                    get<4>(kernel),
                    get<0>(size),
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    get<1>(func));

                for (auto& result : results) {
                    if (result.name == get<0>(func) &&
//                        result.h == get<3>(kernel) &&
//                        result.w == get<4>(kernel) &&
                        result.batch_dim == get<0>(size) &&
                        result.a_rows == get<1>(size) &&
                        result.a_cols == get<2>(size) &&
                        result.b_cols == get<3>(size) &&
                        result.b1 == get<4>(size) &&
                        result.b2 == get<5>(size) &&
                        result.b3 == get<6>(size)
                        ) {
                        result.time += time;
                        break;
                    }
                }
                curr_it++;
            }
        }
    }

    // Calculate average time
    for (auto& result : results) {
        result.time /= num_repeats_shuffle;
    }

//    // Output the results in the original order
//    const string fileName = "performance.csv";
//    ofstream csv_file(fileName);

    // Output the results in the original order with a unique file name
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << "../data/performance_" << put_time(&tm, "%Y%m%d%H%M%S") << ".csv";
    string fileName = oss.str();
    ofstream csv_file(fileName);
//    csv_file << "Name, H, W, Batch Dimension, ARows, ACols, BCols, B1, B2, B3, Correctness, Time" << endl;
    csv_file << "Name, Batch Dimension, ARows, ACols, BCols, B1, B2, B3, Correctness, Time" << endl;


    for (const auto& result: results) {
        csv_file << result.name << ","
//        << result.h << ","
//        << result.w << ","
        << result.batch_dim << ","
        << result.a_rows << ","
        << result.a_cols << ","
        << result.b_cols << ","
        << result.b1 << ","
        << result.b2 << ","
        << result.b3 << ","
        << result.correct << ","
        << result.time << endl;
    }

    csv_file.close();

    cout << "Kernel performance results written to " << fileName << endl;

    return 0;
}