"""
Runs the benchmarking tests for the einsum implementation against numpy's einsum.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
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

# plot results
def plot_results_einsum_str(data_path="benchmark_results.csv", plot_path="benchmark.png"):
    # plot with einsum_str on x-axis
    data = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding=None)
    functions = np.unique(data["function"])
    einsum_strs = np.unique(data["einsum_str"])

    # Initialize arrays to store times for each function
    times_dict = {function: [] for function in functions}

    for function in functions:
        for einsum_str in einsum_strs:
            mask = (data["function"] == function) & (data["einsum_str"] == einsum_str)
            if np.any(mask):
                times_dict[function].append(data["time"][mask][0])
            else:
                times_dict[function].append(0)  # Handle missing data

    x = np.arange(len(einsum_strs))
    width = 0.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Adjust the bar positions based on the number of functions
    for i, (function, times) in enumerate(times_dict.items()):
        ax.bar(x + (i - len(functions) / 2) * width, times, width, label=function)

    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Benchmark: SOME VERY CATCHY TITLE')

    # create format strings for the x-axis labels
    x_labels = [short_str2format_str(einsum_str) for einsum_str in einsum_strs]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    # functions to test with their arguments
    test_functions = {
        "einsum_bmm_naive": (einsum, py_bmm.bmm_naive),
        "einsum_bmm_kernel": (einsum, py_bmm.bmm_kernel),
        "einsum_bmm_parallel": (einsum, py_bmm.bmm_parallel),
        "einsum_bmm_blas": (einsum, py_bmm.bmm_blas),
        "numpy_einsum": (np.einsum,)
    }

    num_repeat = 10

    print("Generating test cases...")
    cases = []
    test_cases = get_testcases()
    # test_cases_bd = get_testcases_with_bd(2)
    # test_cases_only_bd = make_bd_only_testcases([(10, 100, 100, 100), (5, 200, 200, 200)])#, (2, 100, 200, 300), (3, 300, 200, 100)])
    cases.extend(test_cases[1:5])

    print("Running benchmarks...")

    # generate a unique filename for the CSV file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"

    # run benchmarks and save results to CSV
    run_benchmarks(test_functions, cases, num_repeat=num_repeat, data_path=csv_filename)

    # plot results from the CSV file
    plot_results_einsum_str(data_path=csv_filename, plot_path=f"benchmark_{timestamp}.png")