#include <pybind11/pybind11.h>
#include <bmm.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<double> bmm_wrapper(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info A_buf = A.request(), B_buf = B.request();

    if (A_buf.ndim == 3 && B_buf.ndim == 3) {

        if (A_buf.shape[2] != B_buf.shape[1] || A_buf.shape[0] != B_buf.shape[0])
            throw std::runtime_error("Matrix dimensions or batch dimensions do not match for multiplication");

        py::array_t<double> result = py::array_t<double>({A_buf.shape[0], A_buf.shape[1], B_buf.shape[2]});
        py::buffer_info res_buf = result.request();

        double *ptr_A = static_cast<double *>(A_buf.ptr);
        double *ptr_B = static_cast<double *>(B_buf.ptr);
        double *ptr_res = static_cast<double *>(res_buf.ptr);

        int bA = A_buf.shape[0], rA = A_buf.shape[1], cA = A_buf.shape[2], cB = B_buf.shape[2];

        ptr_res = batch_matrix_multiply(ptr_A, ptr_B, ptr_res, bA, rA, cB, cA);

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

        // batch dimension is 1
        ptr_res = batch_matrix_multiply(ptr_A, ptr_B, ptr_res, 1, A_buf.shape[0], B_buf.shape[1], A_buf.shape[1]);

        return result;
    }
    else
        throw std::runtime_error("Number of dimensions must be two or three");

}

PYBIND11_MODULE(py_bmm, m) {
    m.def("bmm", &bmm_wrapper, "multiply two batch matrices");
}