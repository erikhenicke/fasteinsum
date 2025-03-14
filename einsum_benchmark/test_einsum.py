import numpy as np
import pytest
import einsum_bmm

# Set random seed for reproducibility
np.random.seed(0)

@pytest.mark.parametrize("eq, a, b, atol", [
    # Example 1: Matrix multiplication
    ('ij,jk->ik', np.random.rand(2, 3), np.random.rand(3, 4), 1e-6),

    # Example 2: Batch matrix multiplication
    ('bij,bjk->bik', np.random.rand(5, 2, 3), np.random.rand(5, 3, 4), 1e-6),

    # Example 3: Sum over axes
    ('ijk->', np.random.rand(2, 3, 4), None, 1e-6),

    # Example 4: Transpose
    ('ij->ji', np.random.rand(2, 3), None, 1e-6),

    # Example 5: Outer product
    ('i,j->ij', np.random.rand(2), np.random.rand(3), 1e-6),

    # Example 6: Double contraction
    ('ijk,klj->il', np.random.rand(2, 3, 4), np.random.rand(4, 3, 3), 1e-6),

    # Example 7: Triple contraction
    ('ijkl,mlkj->im', np.random.rand(2, 3, 4, 5), np.random.rand(5, 5, 4, 3), 1e-6),

    # Example 8: Complex contraction with multiple indices
    ('ijkl,mnkl->ijmn', np.random.rand(2, 3, 4, 5), np.random.rand(5, 4, 4, 5), 1e-6),

    # Example 9: Contraction with repeated indices
    ('iij->j', np.random.rand(3, 3, 4), None, 1e-6),

    # Example 10: Contraction with summation and transposition
    ('ijk->kji', np.random.rand(2, 3, 4), None, 1e-6),

    # Example 11: Broadcasting with diagonalization
    ('ii,jj->ji', np.random.rand(5, 5), np.random.rand(10, 10), 1e-6),

    # Example 12: Diagonalization, contraction, summation, transposition
    ('iikllj,lkjjo->ij', np.random.rand(5, 5, 8, 6, 6, 10), np.random.rand(6, 8, 10, 10, 12), 1e-6),

    # Example 13: Multiplication
    ('ijk,ikj->ij', np.random.rand(5, 6, 8), np.random.rand(5, 8, 6), 1e-6),
])
def test_einsum(eq, a, b, atol):
    result = einsum_bmm.einsum(eq, a, b) if b is not None else einsum_bmm.einsum(eq, a)
    expected = np.einsum(eq, a, b) if b is not None else np.einsum(eq, a)
    assert np.allclose(result, expected, atol=atol), f"Failed for equation: {eq}"
