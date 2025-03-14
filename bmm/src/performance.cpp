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
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2,
    int b3,
    void (*bmm_wrapper)(const double*, const double*, double*, int, int, int, int, int, int, int, double*))
{
    aligned_vector<double> a(batch_dim * a_rows * a_cols);
    aligned_vector<double> b(batch_dim * a_cols * b_cols);
    aligned_vector<double> c(batch_dim * a_rows * b_cols, 0.0);
    aligned_vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

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
    int batch_dim,
    int a_rows,
    int b_cols,
    int a_cols,
    int b1,
    int b2,
    int b3,
    void (*bmm_wrapper)(const double*, const double*, double*, int, int, int, int, int, int, int, double*))
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
    }
    // Calculate average time
    double total_time = 0.0;
    for (int i = 0; i < num_repeats; ++i) {
        total_time += times[i];
    }
    double avg_time = total_time / num_repeats;

    return avg_time;
}

int main() {
    const int num_repeats_shuffle = 1;

    bool do_correctness_check = true;
    bool do_write_csv = true;

    cout << "Measuring performance..."  << endl;

    /* Contains:
     * 1. BMM name
     * 2. Bmm wrapper function
     */
    vector<
        tuple<
            string,
            void (*)(const double*, const double*, double*, int, int, int, int, int, int, int, double*)>> functions = {
        {"naive", bmm_naive_wrapper},
        {"naive parallel", bmm_naive_parallel_wrapper},
        {"blas", bmm_blas_wrapper},
        {"blas parallel", bmm_blas_parallel_wrapper},
        {"kernel", bmm_kernel_wrapper},
        {"kernel parallel", bmm_kernel_parallel_wrapper},
        {"packing", bmm_packing_wrapper},
        {"packing parallel", bmm_packing_parallel_wrapper},
        {"packing omp", bmm_packing_omp_wrapper},
        {"packing omp parallel", bmm_packing_omp_parallel_wrapper},
        };

    // Contains:
    // 1. Number of repetitions
    // 2. Batch dimension
    // 3. A rows
    // 4. A cols
    // 5. B cols
    // 6. B1
    // 7. B2
    // 8. B3
    vector<tuple<int, int, int, int, int, int, int, int>> sizes = {
        {1, 1, 500, 500, 500, 220, 88, 80},
        };

    const size_t num_configs = sizes.size() * functions.size();

    vector<KernelResult> results;

    cout << "Check correctness: " << endl;
    for (const auto& size : sizes) {
        for (auto& func : functions) {
            bool isCorrect = false;
            if (do_correctness_check) {
                isCorrect = check_correctness(
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    get<7>(size),
                    get<1>(func));
                cout << "\tBMM function: " << get<0>(func) << " " << (isCorrect ? "correct" : "incorrect") << endl;
                }
            else { // skip correctness checks
       			cout << "Skipping correctness checks" << endl;
	            isCorrect = true;
                }

            results.push_back({
                    get<0>(func),
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    get<7>(size),
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
                cout << "\tBMM function: " << get<0>(func) << ", " << get<4>(size) << " (" << curr_it + 1 << "/" << num_configs << ")" << endl;

                double time = measure_performance(
                    get<0>(size),
                    get<1>(size),
                    get<2>(size),
                    get<3>(size),
                    get<4>(size),
                    get<5>(size),
                    get<6>(size),
                    get<7>(size),
                    get<1>(func));

                for (auto& result : results) {
                    if (result.name == get<0>(func) &&
                        result.batch_dim == get<1>(size) &&
                        result.a_rows == get<2>(size) &&
                        result.a_cols == get<3>(size) &&
                        result.b_cols == get<4>(size) &&
                        result.b1 == get<5>(size) &&
                        result.b2 == get<6>(size) &&
                        result.b3 == get<7>(size)
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

    if (do_write_csv) {
        // Output the results in the original order with a unique file name
        auto t = time(nullptr);
        auto tm = *localtime(&t);
        ostringstream oss;
        oss << "../results/performance_" << put_time(&tm, "%Y%m%d%H%M%S") << ".csv";
        string fileName = oss.str();
        ofstream csv_file(fileName);
        csv_file << "Name, Batch Dimension, ARows, ACols, BCols, B1, B2, B3, Correctness, Time" << endl;

        for (const auto& result: results) {
            csv_file << result.name << ","
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
    }

    return 0;
}