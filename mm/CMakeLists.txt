cmake_minimum_required(VERSION 3.10)
project(lib-mm)

# set SOURCE variable to the list of source files
set(SOURCE_FILES src/mm.cpp src/mm_performance.cpp)
set(HEADER_FILES include/mm.h include/mm_performance.h include/aligned_allocator.h)

# Add the library
add_library(mm SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(mm PUBLIC include)

add_executable(benchmark_runner src/benchmark_runner.cpp)
target_link_libraries(benchmark_runner PUBLIC mm)

add_executable(test_kernel src/temp_test_kernel.cpp)
target_link_libraries(test_kernel PUBLIC mm)

add_executable(kernel_optimizer src/kernel_optimizer.cpp)

add_executable(block_optimizer src/block_optimizer.cpp)
target_include_directories(block_optimizer PUBLIC include)
target_link_libraries(block_optimizer PUBLIC mm)


# for now: No python integration
##set(CMAKE_PREFIX_PATH "/home/erik/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
#set(CMAKE_PREFIX_PATH "/home/sonja/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
#set(PYBIND11_FINDPYTHON ON)
#find_package(pybind11 CONFIG REQUIRED)
#
## Add the pybind11 module
#pybind11_add_module(py_bmm src/pybind11_wrapper.cpp)
#target_link_libraries(py_bmm PRIVATE bmm)

# add OMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mm PUBLIC OpenMP::OpenMP_CXX)
endif()


