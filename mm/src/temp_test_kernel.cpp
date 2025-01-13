#include "mm.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    const int size = 256;
    std::vector<double> a(size * size), b(size * size), c(size * size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto &val : a) val = dis(gen);
    for (auto &val : b) val = dis(gen);

    std::fill(c.begin(), c.end(), 0.0);
    mm_naive(a.data(), b.data(), c.data(), size, size, size);
    std::cout << "mm_vectorized_pipe_8 completed" << std::endl;
    // print first 10 elements of c
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;


    std::fill(c.begin(), c.end(), 0.0);
    mm_vectorized_pipe_8(a.data(), b.data(), c.data(), size, size, size);
    std::cout << "mm_vectorized_pipe_8 completed" << std::endl;
    // print first 10 elements of c
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;


    std::fill(c.begin(), c.end(), 0.0);
    mm_kernel(a.data(), b.data(), c.data(), size, size, size);
    std::cout << "mm_kernel completed" << std::endl;
    // print first 10 elements of c
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;


    std::fill(c.begin(), c.end(), 0.0);
    mm_blocked(a.data(), b.data(), c.data(), size, size, size);
    std::cout << "mm_blocked completed" << std::endl;
    // print first 10 elements of c
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}