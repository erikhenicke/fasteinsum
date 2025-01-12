//
// Created by sonja on 10.01.25.
//
#include "mm_performance.h"
#include "mm.h"
#include <chrono>
#include <fstream>
#include <random>
#include <iostream>
#include <vector>

using namespace std;

void run_mm_benchmarks(const std::string &output_file, const std::vector<int> &matrix_sizes, const int num_repeats) {
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    ofs << "Matrix Size,Function,Average Time (ms)" << std::endl;

    std::vector<void (*)(const double *, const double *, double *, const int, const int, const int)> functions = {
        mm_naive, mm_transposed, mm_auto_vectorized, mm_omp_vectorized, mm_vectorized_32, mm_vectorized_64,
        mm_vectorized_pipe_2, mm_vectorized_pipe_8,
    };

    std::vector<std::string> function_names = {
        "mm_naive", "mm_transposed", "mm_auto_vectorized", "mm_omp_vectorized", "mm_vectorized_32", "mm_vectorized_64",
        "mm_vectorized_pipe_2", "mm_vectorized_pipe_8",
    };

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int size : matrix_sizes) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<double> a(size * size), b(size * size), c(size * size);

        for (auto &val : a) val = dis(gen);
        for (auto &val : b) val = dis(gen);

        for (size_t i = 0; i < functions.size(); ++i) {
            double total_time = 0.0;

            for (int repeat = 0; repeat < num_repeats; ++repeat) {
                std::fill(c.begin(), c.end(), 0.0);

                auto start = std::chrono::high_resolution_clock::now();
                functions[i](a.data(), b.data(), c.data(), size, size, size);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> elapsed = end - start;
                total_time += elapsed.count();
            }

            double average_time = total_time / num_repeats;
            ofs << size << "," << function_names[i] << "," << average_time << std::endl;
        }
    }

    ofs.close();
}