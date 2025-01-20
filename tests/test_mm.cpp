#include "catch.hpp"
#include "mm.h"
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

bool compare_matrices(const vector<double> &a, const vector<double> &b) {
    const double epsilon = 1e-6;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

void generate_random_matrix(vector<double> &matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void test_function(void (*func)(const double*, const double*, double*, int, int, int), int size) {
    vector<double> a(size * size), b(size * size), c(size * size), c_ref(size * size);
    generate_random_matrix(a, size, size);
    generate_random_matrix(b, size, size);

    mm_naive(a.data(), b.data(), c_ref.data(), size, size, size);
    func(a.data(), b.data(), c.data(), size, size, size);

    REQUIRE(compare_matrices(c, c_ref));
}

TEST_CASE("Matrix Multiplication Functions", "[mm]") {
    const int sizes[] = {64, 128, 256, 512};

    SECTION("mm_naive") {
        for (int size : sizes) {
            test_function(mm_naive, size);
        }
    }

    SECTION("mm_transposed") {
        for (int size : sizes) {
            test_function(mm_transposed, size);
        }
    }

    SECTION("mm_auto_vectorized") {
      for (int size : sizes) {
          test_function(mm_auto_vectorized, size);
      }
    }

    SECTION("mm_omp_vectorized") {
      for (int size : sizes) {
          test_function(mm_omp_vectorized, size);
      }
    }

    SECTION("mm_vectorized_32") {
        for (int size : sizes) {
            test_function(mm_vectorized_32, size);
        }
    }

    SECTION("mm_vectorized_64") {
        for (int size : sizes) {
            test_function(mm_vectorized_64, size);
        }
    }

    SECTION("mm_vectorized_pipe_2") {
        for (int size : sizes) {
            test_function(mm_vectorized_pipe_2, size);
        }
    }

    SECTION("mm_vectorized_pipe_8") {
        for (int size : sizes) {
            test_function(mm_vectorized_pipe_8, size);
        }
    }
    SECTION("mm_kernel") {
        for (int size : sizes) {
            test_function(mm_kernel, size);
        }
    }
    SECTION("mm_blocked") {
        for (int size : sizes) {
            test_function(mm_blocked, size);
        }
    }
    SECTION("mm_blocked_packed") {
        for (int size : sizes) {
            test_function(mm_blocked_packed, size);
        }
    }
}