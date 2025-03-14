# some tests for einsum_bmm.py

import numpy as np
from einsum_bmm import einsum

np.random.seed(0)

# Example 1: Matrix multiplication
a = np.random.rand(2, 3)
b = np.random.rand(3, 4)
result = einsum('ij,jk->ik', a, b)
result_np = np.einsum('ij,jk->ik', a, b)
assert np.allclose(result, result_np)

# Example 2: Batch matrix multiplication
a = np.random.rand(5, 2, 3)
b = np.random.rand(5, 3, 4)
result = einsum('bij,bjk->bik', a, b)
result_np = np.einsum('bij,bjk->bik', a, b)
assert np.allclose(result, result_np)

# Example 3: Sum over axes
a = np.random.rand(2, 3, 4)
result = einsum('ijk->', a)
result_np = np.einsum('ijk->', a)
assert np.allclose(result, result_np)

# Example 4: Transpose
a = np.random.rand(2, 3)
result = einsum('ij->ji', a)
result_np = np.einsum('ij->ji', a)
assert np.allclose(result, result_np)

# Example 5: Outer product
a = np.random.rand(2)
b = np.random.rand(3)
result = einsum('i,j->ij', a, b)
result_np = np.einsum('i,j->ij', a, b)
assert np.allclose(result, result_np)

# Example 6: Double contraction
a = np.random.rand(2, 3, 4)
b = np.random.rand(4, 3, 3)
result = einsum('ijk,klj->il', a, b)
result_np = np.einsum('ijk,klj->il', a, b)
assert np.allclose(result, result_np)

# Example 7: Triple contraction
a = np.random.rand(2, 3, 4, 5)
b = np.random.rand(5, 5, 4, 3)
result = einsum('ijkl,mlkj->im', a, b)
result_np = np.einsum('ijkl,mlkj->im', a, b)
assert np.allclose(result, result_np)

# Example 8: Complex contraction with multiple indices
a = np.random.rand(2, 3, 4, 5)
b = np.random.rand(5, 4, 4, 5)
result = einsum('ijkl,mnkl->ijmn', a, b)
result_np = np.einsum('ijkl,mnkl->ijmn', a, b)
assert np.allclose(result, result_np)

# Example 9: Contraction with repeated indices
a = np.random.rand(3, 3, 4)
result = einsum('iij->j', a)
result_np = np.einsum('iij->j', a)
assert np.allclose(result, result_np)

# Example 10: Contraction with summation and transposition
a = np.random.rand(2, 3, 4)
result = einsum('ijk->kji', a)
result_np = np.einsum('ijk->kji', a)
assert np.allclose(result, result_np)

# Example 11: Broadcasting with diagonalisation
a = np.random.rand(5, 5)
b = np.random.rand(10, 10)
result = einsum('ii,jj->ji', a, b)
result_np = np.einsum('ii,jj->ji', a, b)
assert np.allclose(result, result_np)

# Example 12: Diagonalisation, contraction, summation, transposition
a = np.random.rand(5, 5, 8, 6, 6, 10)
b = np.random.rand(6, 8, 10, 10, 12)
result = einsum('iikllj,lkjjo->ij', a, b)
result_np = np.einsum('iikllj,lkjjo->ij', a, b)
assert np.allclose(result, result_np)

# Example 13: Multiplication
a = np.random.rand(5, 6, 8)
b = np.random.rand(5, 8, 6)
result = einsum('ijk,ikj->ij', a, b)
result_np = np.einsum('ijk,ikj->ij', a, b)
assert np.allclose(result, result_np)
