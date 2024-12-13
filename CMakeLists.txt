cmake_minimum_required(VERSION 3.10)
project(einsum-bmm)

add_subdirectory(bmm)
add_subdirectory(integration)
add_subdirectory(catch)
add_subdirectory(tests)

enable_testing()
add_test(NAME CATCH_BMM_TESTS COMMAND catch_tests_bmm)