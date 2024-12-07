cmake_minimum_required(VERSION 3.10)
project(bmm)

# Add the example library
add_library(hello_bmm SHARED src/hello_bmm.cpp include/hello_bmm.h)
target_include_directories(hello_bmm PUBLIC include)

set(CMAKE_PREFIX_PATH "/home/erik/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(pybind11_example src/pybind11_wrapper.cpp)
target_link_libraries(pybind11_example PRIVATE hello_bmm)


