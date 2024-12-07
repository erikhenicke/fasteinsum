#include <pybind11/pybind11.h>
#include <hello_bmm.h>

PYBIND11_MODULE(pybind11_example, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    m.def("cpp_function", &hello_bmm, "A function that multiplies two numbers");
}