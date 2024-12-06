cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(hello_bmm LANGUAGES CXX)


add_executable(hello_bmm)
target_sources(hello_bmm PUBLIC main.cpp)
#target_compile_options(quantpng PUBLIC -O2 -fopenmp)
#target_link_libraries(quantpng PUBLIC -fopenmp)

add_subdirectory(bmm_lib_example)

# Link lodepng and quantize libraries to the target
target_link_libraries(hello_bmm PUBLIC bmm_lib)