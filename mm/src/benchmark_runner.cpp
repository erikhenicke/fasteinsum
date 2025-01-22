//
// Created by sonja on 10.01.25.
//
#include "mm_performance.h"
#include <iostream>
#include <vector>

int main() {
    std::string output_file = "benchmark_results.csv";
    std::vector<int> matrix_sizes = {1024, 2048};
    int num_repeats = 3;

    run_mm_benchmarks(output_file, matrix_sizes, num_repeats);

    std::cout << "Benchmark results saved to " << output_file << std::endl;
    return 0;
}