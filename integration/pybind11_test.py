import pybind11_example
import numpy as np

if __name__ == "__main__":

    a = np.random.rand(6)
    b = np.random.rand(6)

    result = pybind11_example.add_arrays(a, b)
    print(f"a:   {a}\nb:   {b}\nres: {result}")
