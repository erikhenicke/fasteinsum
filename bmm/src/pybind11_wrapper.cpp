#include <pybind11/pybind11.h>
#include <hello_bmm.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> wrapper(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    ptr3 = add_arrays(ptr1, ptr2, ptr3, buf1.size);

    return result;
}

PYBIND11_MODULE(pybind11_example, m) {
    m.def("add_arrays", &wrapper, "Add two NumPy arrays");
}