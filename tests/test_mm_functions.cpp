#include <iostream>
#include <vector>
#include "mm.h"

using namespace std;

void print_matrix(const vector<double> &matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

void multiply_and_print(const double *a, const double *b, double *c, int a_rows, int b_cols, int a_cols,
                        void (*mm_func)(const double*, const double*, double*, int, int, int), const string &func_name) {
    // Reset c to zero
    fill(c, c + a_rows * b_cols, 0.0);
    mm_func(a, b, c, a_rows, b_cols, a_cols);
    cout << "Result of " << func_name << ":" << endl;
    print_matrix(vector<double>(c, c + a_rows * b_cols), a_rows, b_cols);
    cout << endl;
}

int main() {
    const int a_rows = 3, a_cols = 3, b_cols = 3;
    vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<double> b = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    vector<double> c(a_rows * b_cols, 0.0);

    print_matrix(a, a_rows, a_cols);
    print_matrix(b, a_cols, b_cols);
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_naive, "mm_naive");

    print_matrix(a, a_rows, a_cols);
    print_matrix(b, a_cols, b_cols);
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_transposed, "mm_transposed");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_auto_vectorized, "mm_auto_vectorized");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_omp_vectorized, "mm_omp_vectorized");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_vectorized_32, "mm_vectorized_32");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_vectorized_64, "mm_vectorized_64");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_vectorized_pipe_2, "mm_vectorized_pipe_2");
    multiply_and_print(a.data(), b.data(), c.data(), a_rows, b_cols, a_cols, mm_vectorized_pipe_8, "mm_vectorized_pipe_8");

    return 0;
}