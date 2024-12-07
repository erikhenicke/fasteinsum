import py_bmm
import numpy as np

if __name__ == "__main__":

    a = np.random.rand(2, 4, 8)
    b = np.random.rand(2, 8, 4)

    res1 = py_bmm.bmm(a, b)
    print(f"a:    {a}\nb:    {b}\nres1: {res1}")

    res2 = np.einsum("bij,bjk->bik", a, b)
    print(f"res2: {res2}")
