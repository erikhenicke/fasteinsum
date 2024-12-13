cmake_minimum_required(VERSION 3.10)
project(catch_tests_bmm LANGUAGES CXX)

#set standards compiler independent
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#create an executable with the test cases
add_executable(catch_tests_bmm catch_main.cpp catch_tests_bmm.cpp)

# link libraries fizzbuzz and catch to the executable "fizzbuzz_test"
target_link_libraries(catch_tests_bmm PUBLIC bmm catch)

add_test(NAME CATCH_BMM_TESTS COMMAND catch_tests_bmm)