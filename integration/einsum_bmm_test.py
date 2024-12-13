# some tests for einsum_bmm.py

import numpy as np
from einsum_bmm import einsum

# Example 1: Matrix multiplication
a = np.random.rand(2, 3)
b = np.random.rand(3, 4)
result = einsum('ij,jk->ik', a, b)
print("Matrix multiplication result:\n", result)

# Example 2: Batch matrix multiplication
a = np.random.rand(5, 2, 3)
b = np.random.rand(5, 3, 4)
result = einsum('bij,bjk->bik', a, b)
print("Batch matrix multiplication result:\n", result)

# Example 3: Sum over axes
a = np.random.rand(2, 3, 4)
result = einsum('ijk->', a)
print("Sum over all elements result:\n", result)

# Example 4: Transpose
a = np.random.rand(2, 3)
result = einsum('ij->ji', a)
print("Transpose result:\n", result)

# Example 5: Outer product
a = np.random.rand(2)
b = np.random.rand(3)
result = einsum('i,j->ij', a, b)
print("Outer product result:\n", result)


# Example 6: Double contraction
a = np.random.rand(2, 3, 4)
b = np.random.rand(4, 3, 3)
result = einsum('ijk,klj->il', a, b)
print("Double contraction result:\n", result)

# Example 7: Triple contraction
a = np.random.rand(2, 3, 4, 5)
b = np.random.rand(5, 5, 4, 3)
result = einsum('ijkl,mlkj->im', a, b)
print("Triple contraction result:\n", result)

# Example 8: Complex contraction with multiple indices
a = np.random.rand(2, 3, 4, 5)
b = np.random.rand(5, 4, 4, 5)
result = einsum('ijkl,mnkl->ijmn', a, b)
print("Complex contraction result:\n", result)

# Example 9: Contraction with repeated indices
a = np.random.rand(3, 3, 4)
result = einsum('iij->j', a)
print("Contraction with repeated indices result:\n", result)

# Example 10: Contraction with summation and transposition
a = np.random.rand(2, 3, 4)
result = einsum('ijk->kji', a)
print("Contraction with summation and transposition result:\n", result)