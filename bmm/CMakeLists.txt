cmake_minimum_required(VERSION 3.10)
project(lib-bmm)

find_package(BLAS REQUIRED)

# set SOURCE variable to the list of source files
set(SOURCE_FILES src/bmm.cpp src/kernels.cpp src/wrapper.cpp)
set(HEADER_FILES include/bmm.h include/aligned_allocator.h include/kernels.h include/wrapper.h)

# Add the example library
add_library(bmm SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(bmm PUBLIC include)
target_link_libraries(bmm PUBLIC ${BLAS_LIBRARIES})

#add_executable(bmm_debug src/bmm.cpp)
#target_include_directories(bmm_debug PUBLIC include)
#target_link_libraries(bmm_debug PUBLIC bmm ${BLAS_LIBRARIES})
#
#add_executable(kernels src/kernels.cpp)
#target_include_directories(kernels PUBLIC include)
#target_link_libraries(kernels PUBLIC bmm ${BLAS_LIBRARIES})

add_executable(performance src/performance.cpp)
target_include_directories(performance PUBLIC include)
target_link_libraries(performance PUBLIC bmm)

set(CMAKE_PREFIX_PATH "/home/erik/.venv_bmm/lib/python3.13/site-packages/pybind11/share/cmake/pybind11/")
#set(CMAKE_PREFIX_PATH "/home/sonja/.bmm_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
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


