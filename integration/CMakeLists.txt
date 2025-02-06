# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set PYTHONPATH to include the build directory
# set(PYTHONPATH "${CMAKE_BINARY_DIR}/bmm")
set(PYTHONPATH "${CMAKE_BINARY_DIR}/bmm:$ENV{PYTHONPATH}")
message("Python exacutable: ${Python3_EXECUTABLE}")
#PYTHONPATH=/home/sonja/.bmm_venv/lib/python3.12/site-packages:/home/sonja/Alg_Eng/ae_ws2024_bmm/build/bmm
#/home/sonja/.bmm_venv/bin/python

# Add python test as target
add_custom_target(bmm_test
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${Python3_EXECUTABLE} bmm_test.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS py_bmm
)

add_custom_target(einsum_bmm_test
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${Python3_EXECUTABLE} einsum_bmm_test.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS py_bmm
)

add_custom_target(benchmark
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${Python3_EXECUTABLE} benchmark.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS py_bmm
)