cmake_minimum_required(VERSION 3.10)
project(einsum-bmm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -fopenmp")

add_subdirectory(bmm)
add_subdirectory(catch)
add_subdirectory(tests)

enable_testing()
add_test(NAME CATCH_BMM_TESTS COMMAND catch_tests_bmm)

