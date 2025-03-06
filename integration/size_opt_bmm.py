import py_bmm

def size_opt_bmm(a, b):
    """
    Uses naive bmm for small matrices and bmm_parallel for large matrices

    :param a: tensor of shape (bd, m, n)
    :param b: tensor of shape (bd, n, p)
    :return: c = a @ b, tensor of shape (bd, m, p)
    """
    bd, m, n = a.shape
    bd, n, p = b.shape
    if bd = 1 and m <= 100 and n <= 100 and p <= 100:
        return py_bmm.bmm_naive(a, b)
    else:
        return py_bmm.bmm_parallel(a, b)