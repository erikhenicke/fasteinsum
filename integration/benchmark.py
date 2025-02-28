"""
Runs the benchmarking tests for the einsum implementation against numpy's einsum.
"""

import time
import numpy as np
# import matplotlib.pyplot as plt
import py_bmm
from einsum_bmm import einsum
from benchmark_set import get_testcases, get_testcases_with_bd, make_bd_only_testcases, generate_einsum_input

# functions to test with their arguments
test_functions = {
    "einsum_bmm_naive": (einsum, py_bmm.bmm_naive),
    "einsum_bmm": (einsum, py_bmm.bmm),
    "einsum_bmm_parallel": (einsum, py_bmm.bmm_parallel),
    "numpy_einsum": (np.einsum,)
}

# get an example test case and ruin it for all functions
test_cases = get_testcases()
test_cases_bd = get_testcases_with_bd(2)
test_cases_only_bd = make_bd_only_testcases([(10, 100, 100, 100)])

# check size of test cases and if they even can be generated
for test_case in test_cases + test_cases_bd + test_cases_only_bd:
    einsum_str, tensor1, tensor2 = generate_einsum_input(test_case)
    if tensor1 is not None and tensor2 is not None:
        print(f"Example input for einsum: {einsum_str}, {tensor1.shape}, {tensor2.shape}")




idx= 2;

for func_name, (func, *args) in test_functions.items():
    print(f"Running benchmark for {func_name}")
    for test_case in test_cases[:idx]:
        einsum_str, tensor1, tensor2 = generate_einsum_input(test_case)
        if tensor1 is None or tensor2 is None:
            continue
        start = time.time()
        result = func(einsum_str, tensor1, tensor2)
        end = time.time()
        print(f"Time for {func_name} with test case {test_case}: {end - start}")

    for test_case in test_cases_bd[:idx]:
        einsum_str, tensor1, tensor2 = generate_einsum_input(test_case)
        if tensor1 is None or tensor2 is None:
            continue
        start = time.time()
        result = func(einsum_str, tensor1, tensor2)
        end = time.time()
        print(f"Time for {func_name} with test case {test_case}: {end - start}")

    for test_case in test_cases_only_bd:
        einsum_str, tensor1, tensor2 = generate_einsum_input(test_case)
        if tensor1 is None or tensor2 is None:
            continue
        start = time.time()
        result = func(einsum_str, tensor1, tensor2)
        end = time.time()
        print(f"Time for {func_name} with test case {test_case}: {end - start}")

    print("")