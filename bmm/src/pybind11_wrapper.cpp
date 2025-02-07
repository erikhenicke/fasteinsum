#include <pybind11/pybind11.h>
#include <bmm.h>
#include <iostream>
#include <kernels.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<double> bmm_wrapper(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info A_buf = A.request(), B_buf = B.request();

    std::cout << "Bmm fast" << std::endl;

    int h = 8;
    int w = 16;
    int simd_length = 4;
    int wl = w / simd_length;
    int b1 = 32;
    int b2_ = 64;
    int b3_ = 128;

    if (A_buf.ndim == 3 && B_buf.ndim == 3) {

        if (A_buf.shape[2] != B_buf.shape[1] || A_buf.shape[0] != B_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        py::array_t<double> result = py::array_t<double>({A_buf.shape[0], A_buf.shape[1], B_buf.shape[2]});
        py::buffer_info res_buf = result.request();

        double *ptr_A = static_cast<double *>(A_buf.ptr);
        double *ptr_B = static_cast<double *>(B_buf.ptr);
        double *ptr_res = static_cast<double *>(res_buf.ptr);

        int bA = A_buf.shape[0], rA = A_buf.shape[1], cA = A_buf.shape[2], cB = B_buf.shape[2];

        bmm(ptr_A, ptr_B, ptr_res, bA, rA, cB, cA, h, w, simd_length, wl, b1, b2_, b3_, kernel_8x16);

        return result;
        }

    else if (A_buf.ndim == 2 && B_buf.ndim == 2) {
        if (A_buf.shape[1] != B_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        py::array_t<double> result = py::array_t<double>({A_buf.shape[0], B_buf.shape[1]});
        py::buffer_info res_buf = result.request();

        double *ptr_A = static_cast<double *>(A_buf.ptr);
        double *ptr_B = static_cast<double *>(B_buf.ptr);
        double *ptr_res = static_cast<double *>(res_buf.ptr);

        int rA = A_buf.shape[0], cA = A_buf.shape[1], cB = B_buf.shape[1];

        // Batch dimension is 1
        bmm(ptr_A, ptr_B, ptr_res, 1, rA, cB, cA, h, w, simd_length, wl, b1, b2_, b3_, kernel_8x16);

        return result;
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

}

py::array_t<double> bmm_naive_wrapper(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info A_buf = A.request(), B_buf = B.request();

    std::cout << "Bmm naive" << std::endl;

    if (A_buf.ndim == 3 && B_buf.ndim == 3) {

        if (A_buf.shape[2] != B_buf.shape[1] || A_buf.shape[0] != B_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        py::array_t<double> result = py::array_t<double>({A_buf.shape[0], A_buf.shape[1], B_buf.shape[2]});
        py::buffer_info res_buf = result.request();

        double *ptr_A = static_cast<double *>(A_buf.ptr);
        double *ptr_B = static_cast<double *>(B_buf.ptr);
        double *ptr_res = static_cast<double *>(res_buf.ptr);

        int bA = A_buf.shape[0], rA = A_buf.shape[1], cA = A_buf.shape[2], cB = B_buf.shape[2];

        ptr_res = bmm_naive(ptr_A, ptr_B, ptr_res, bA, rA, cB, cA);

        return result;
        }

    else if (A_buf.ndim == 2 && B_buf.ndim == 2) {
        if (A_buf.shape[1] != B_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        py::array_t<double> result = py::array_t<double>({A_buf.shape[0], B_buf.shape[1]});
        py::buffer_info res_buf = result.request();

        double *ptr_A = static_cast<double *>(A_buf.ptr);
        double *ptr_B = static_cast<double *>(B_buf.ptr);
        double *ptr_res = static_cast<double *>(res_buf.ptr);

        int rA = A_buf.shape[0], cA = A_buf.shape[1], cB = B_buf.shape[1];

        // batch dimension is 1
        ptr_res = bmm_naive(ptr_A, ptr_B, ptr_res, 1, rA, cB, cA);

        return result;
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");
}

PYBIND11_MODULE(py_bmm, m) {
    m.def("bmm", &bmm_wrapper, "multiply two batch matrices");
    m.def("bmm_naive", &bmm_naive_wrapper, "multiply two batch matrices");
}