cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(hello_bmm LANGUAGES CXX)

add_executable(hello_bmm)
target_sources(hello_bmm PUBLIC ${PROJECT_SOURCE_DIR}/main.cpp)

add_subdirectory(bmm_lib_example)

target_link_libraries(hello_bmm PUBLIC bmm_lib)