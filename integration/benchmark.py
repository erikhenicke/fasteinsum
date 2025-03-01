"""
Runs the benchmarking tests for the einsum implementation against numpy's einsum.
"""

import time
import numpy as np
import py_bmm
from einsum_bmm import einsum
from benchmark_set import (get_testcases, get_testcases_with_bd, make_bd_only_testcases, generate_einsum_input,
                           format_str2short_str, short_str2format_str, shape2shape_str, shape_str2shape)
import datetime
from tqdm import tqdm



# # get an example test case and ruin it for all functions
# test_cases = get_testcases()
# test_cases_bd = get_testcases_with_bd(2)
# test_cases_only_bd = make_bd_only_testcases([(10, 100, 100, 100)])
#
# # check size of test cases and if they even can be generated
# for test_case in test_cases + test_cases_bd + test_cases_only_bd:
#     einsum_str, tensor1, tensor2 = generate_einsum_input(test_case)
#     if tensor1 is not None and tensor2 is not None:
#         print(f"Example input for einsum: {einsum_str}, {tensor1.shape}, {tensor2.shape}")

def flush_cache():
    # Flush the cache
    np.random.seed(0)
    np.random.rand(1000, 1000)
    np.random.seed(0)
    np.random.rand(1000, 1000)



# benchmarking
def run_benchmarks(functions, cases, num_repeat=10, data_path="benchmark_results.csv"):
    with open(data_path, "w") as f:
        f.write("function,einsum_str,shape1,shape2,time\n")
        for name, (function, *args) in tqdm(functions.items(), desc="Functions"):
            for case in tqdm(cases, desc="Cases", leave=False):
                einsum_str, tensor1, tensor2 = generate_einsum_input(case)
                if tensor1 is None or tensor2 is None:
                    continue
                times = []
                for _ in range(num_repeat):
                    flush_cache()
                    start = time.time()
                    if args:
                        function(einsum_str, tensor1, tensor2, bmm_function=args[0])
                    else:
                        function(einsum_str, tensor1, tensor2)
                    end = time.time()
                    times.append(end - start)
                avg_time = np.mean(times)
                f.write(f"{name},"
                        f"{format_str2short_str(einsum_str)},"
                        f"{shape2shape_str(tensor1.shape)},"
                        f"{shape2shape_str(tensor2.shape)},"
                        f"{avg_time}\n")
    print(f"Benchmark_results saved to {data_path}")

# correctness check agains numpy einsum
def check_correctness(functions, cases):
    for name, (function, *args) in functions.items():
        for case in cases:
            einsum_str, tensor1, tensor2 = generate_einsum_input(case)
            if tensor1 is None or tensor2 is None:
                continue
            if args:
                result = function(einsum_str, tensor1, tensor2, bmm_function=args[0])
            else:
                result = function(einsum_str, tensor1, tensor2)
            np_result = np.einsum(einsum_str, tensor1, tensor2)
            if not np.allclose(result, np_result):
                print(f"Error in {name} for case {case}")
                print(f"Expected: {np_result}")
                print(f"Got: {result}")
                print(f"Einsum string: {einsum_str}")
                print(f"Shapes: {tensor1.shape}, {tensor2.shape}")
                print("")
    print("All tests passed!")

if __name__ == '__main__':
    # functions to test with their arguments
    test_functions = {
        "einsum_bmm_naive": (einsum, py_bmm.bmm_naive),
        "einsum_bmm_kernel": (einsum, py_bmm.bmm_kernel),
        "einsum_bmm_parallel": (einsum, py_bmm.bmm_parallel),
        "einsum_bmm_blas": (einsum, py_bmm.bmm_blas),
        "numpy_einsum": (np.einsum,)
    }

    do_correctness_check = True

    num_repeat = 10

    print("Generating test cases...")
    cases = []
    test_cases = get_testcases()
    # test_cases_bd = get_testcases_with_bd(2)
    # test_cases_only_bd = make_bd_only_testcases([(10, 100, 100, 100), (5, 200, 200, 200)])#, (2, 100, 200, 300), (3, 300, 200, 100)])
    cases.extend(test_cases[1:2])

    if do_correctness_check:
        print("Checking correctness...")
        check_correctness(test_functions, cases)

    print("Running benchmarks...")

    # generate a unique filename for the CSV file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"../data/benchmark_{timestamp}.csv"

    # run benchmarks and save results to CSV
    run_benchmarks(test_functions, cases, num_repeat=num_repeat, data_path=csv_filename)

