#include <pybind11/pybind11.h>
#include <bmm.h>
#include <iostream>
#include <kernels.h>
#include <wrapper.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<double> bmm_avx2_pywrapper(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();

    double *ptr_a = static_cast<double *>(a_buf.ptr);
    double *ptr_b = static_cast<double *>(b_buf.ptr);
    py::array_t<double> result;
    double *ptr_res;
    int bd, a_rows, a_cols, b_cols;

    int h = 8;
    int w = 16;
    int b1 = 220;
    int b2_ = 88; // 88 % 8  = 0
    int b3_ = 80; // 80 % 16 = 0

    if (a_buf.ndim == 3 && b_buf.ndim == 3) {
        if (a_buf.shape[2] != b_buf.shape[1] || a_buf.shape[0] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = a_buf.shape[0], a_rows = a_buf.shape[1], a_cols = a_buf.shape[2], b_cols = b_buf.shape[2];
    }

    else if (a_buf.ndim == 2 && b_buf.ndim == 2) {
        if (a_buf.shape[1] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], b_buf.shape[1]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = 1, a_rows = a_buf.shape[0], a_cols = a_buf.shape[1], b_cols = b_buf.shape[1];
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

    bmm_packing_parallel(ptr_a, ptr_b, ptr_res, bd, a_rows, b_cols, a_cols, h, w, b1, b2_, b3_,
                        kernel_8x16_pack_local_c);

    return result;
}

py::array_t<double> bmm_omp_pywrapper(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();

    double *ptr_a = static_cast<double *>(a_buf.ptr);
    double *ptr_b = static_cast<double *>(b_buf.ptr);
    py::array_t<double> result;
    double *ptr_res;
    int bd, a_rows, a_cols, b_cols;

    int h = 8;
    int w = 16;
    int b1 = 220;
    int b2_ = 88; // 88 % 8  = 0
    int b3_ = 80; // 80 % 16 = 0

    if (a_buf.ndim == 3 && b_buf.ndim == 3) {
        if (a_buf.shape[2] != b_buf.shape[1] || a_buf.shape[0] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = a_buf.shape[0], a_rows = a_buf.shape[1], a_cols = a_buf.shape[2], b_cols = b_buf.shape[2];
    }
    else if (a_buf.ndim == 2 && b_buf.ndim == 2) {
        if (a_buf.shape[1] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], b_buf.shape[1]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = 1, a_rows = a_buf.shape[0], a_cols = a_buf.shape[1], b_cols = b_buf.shape[1];
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

    bmm_packing_parallel(ptr_a, ptr_b, ptr_res, bd, a_rows, b_cols, a_cols, h, w, b1, b2_, b3_,
                    kernel_8x16_pack_local_c_omp);

    return result;

}

py::array_t<double> bmm_naive_pywrapper(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();

    double *ptr_a = static_cast<double *>(a_buf.ptr);
    double *ptr_b = static_cast<double *>(b_buf.ptr);
    py::array_t<double> result;
    double *ptr_res;
    int bd, a_rows, a_cols, b_cols;

    if (a_buf.ndim == 3 && b_buf.ndim == 3) {
        if (a_buf.shape[2] != b_buf.shape[1] || a_buf.shape[0] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = a_buf.shape[0], a_rows = a_buf.shape[1], a_cols = a_buf.shape[2], b_cols = b_buf.shape[2];
    }

    else if (a_buf.ndim == 2 && b_buf.ndim == 2) {
        if (a_buf.shape[1] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], b_buf.shape[1]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = 1, a_rows = a_buf.shape[0], a_cols = a_buf.shape[1], b_cols = b_buf.shape[1];
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

    bmm_naive_parallel(ptr_a, ptr_b, ptr_res, bd, a_rows, b_cols, a_cols);

    return result;

}

py::array_t<double> bmm_blas_pywrapper(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();

    double *ptr_a = static_cast<double *>(a_buf.ptr);
    double *ptr_b = static_cast<double *>(b_buf.ptr);
    py::array_t<double> result;
    double *ptr_res;
    int bd, a_rows, a_cols, b_cols;

    if (a_buf.ndim == 3 && b_buf.ndim == 3) {
        if (a_buf.shape[2] != b_buf.shape[1] || a_buf.shape[0] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = a_buf.shape[0], a_rows = a_buf.shape[1], a_cols = a_buf.shape[2], b_cols = b_buf.shape[2];
    }
    else if (a_buf.ndim == 2 && b_buf.ndim == 2) {
        if (a_buf.shape[1] != b_buf.shape[0])
            throw std::runtime_error("Matrix dimensions do not match for multiplication");

        result = py::array_t<double>({a_buf.shape[0], b_buf.shape[1]});
        py::buffer_info res_buf = result.request();
        ptr_res = static_cast<double *>(res_buf.ptr);
        bd = 1, a_rows = a_buf.shape[0], a_cols = a_buf.shape[1], b_cols = b_buf.shape[1];
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

    // batch dimension is 1
    bmm_blas(ptr_a, ptr_b, ptr_res, bd, a_rows, b_cols, a_cols);

    return result;
}

PYBIND11_MODULE(py_bmm, m) {
    m.def("bmm_avx2", &bmm_avx2_pywrapper, "Bmm with packing and 8x16 kernel using AVX2");
    m.def("bmm_omp", &bmm_omp_pywrapper, "BMM with packing, 8x16 kernel and parallelization using only OMP");
    m.def("bmm_naive", &bmm_naive_pywrapper, "BMM naive implementation with parallelization");
    m.def("bmm_blas", &bmm_blas_pywrapper, "BMM using BLAS");
}