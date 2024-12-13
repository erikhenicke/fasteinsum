cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(catch LANGUAGES CXX)

# INTERFACE - use for header only libraries like in video on catch2
add_library(catch INTERFACE)
target_include_directories(catch INTERFACE include)
target_compile_features(catch INTERFACE cxx_std_11)
