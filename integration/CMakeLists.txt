# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set PYTHONPATH to include the build directory
set(PYTHONPATH "${CMAKE_BINARY_DIR}/bmm")

# Add python test as target
add_custom_target(bmm_test
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${Python3_EXECUTABLE} bmm_test.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS py_bmm
)