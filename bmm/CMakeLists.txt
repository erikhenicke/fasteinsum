cmake_minimum_required(VERSION 3.10)
project(lib-bmm)

# set SOURCE variable to the list of source files
set(SOURCE_FILES src/bmm.cpp src/bmm_blocking.cpp src/bmm_omp.cpp)
set(HEADER_FILES include/bmm.h include/bmm_blocking.h include/bmm_omp.h)

# Add the example library
add_library(bmm SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(bmm PUBLIC include)

#set(CMAKE_PREFIX_PATH "/home/erik/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
set(CMAKE_PREFIX_PATH "/home/sonja/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(py_bmm src/pybind11_wrapper.cpp)
target_link_libraries(py_bmm PRIVATE bmm)

# add OMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bmm PUBLIC OpenMP::OpenMP_CXX)
endif()


