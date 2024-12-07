cmake_minimum_required(VERSION 3.10)
project(lib-bmm)

# Add the example library
add_library(bmm SHARED src/bmm.cpp include/bmm.h)
target_include_directories(bmm PUBLIC include)

set(CMAKE_PREFIX_PATH "/home/erik/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(py_bmm src/pybind11_wrapper.cpp)
target_link_libraries(py_bmm PRIVATE bmm)


