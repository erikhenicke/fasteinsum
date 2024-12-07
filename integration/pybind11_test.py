import pybind11_example

if __name__ == "__main__":
    x, y = 6, 2.3
    result = pybind11_example.cpp_function(x, y)
    print(f"In python the result is: {result}")
