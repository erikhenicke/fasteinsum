cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(bmm_lib LANGUAGES CXX)

# List all source files and headers
set(SOURCES
        src/hello_bmm.cpp
)

set(HEADERS
        include/hello_bmm.h
)

# Add the library
add_library(bmm_lib STATIC ${SOURCES} ${HEADERS})
target_include_directories(bmm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)


