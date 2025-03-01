# Make nice plots from raw data in csv files
import time
import numpy as np
import matplotlib.pyplot as plt

from einsum_bmm import einsum
from benchmark_set import format_str2short_str, short_str2format_str, shape2shape_str, shape_str2shape


def plot_einsum_str_time(data_path="benchmark_results.csv", plot_path="benchmark.png"):
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


def plot_matrix_size_time(data_path="benchmark_results.csv", plot_path="benchmark.png"):
    # plot with matrix size on x-axis
    data = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding=None)
    functions = np.unique(data["function"])
    shapes1 = np.unique(data["shape1"])
    shapes2 = np.unique(data["shape2"])

    # Initialize arrays to store times for each function
    times_dict = {function: [] for function in functions}

    for function in functions:
        for shape1, shape2 in zip(shapes1, shapes2):
            mask = (data["function"] == function) & (data["shape1"] == shape1) & (data["shape2"] == shape2)
            if np.any(mask):
                times_dict[function].append(data["time"][mask][0])
            else:
                times_dict[function].append(0)  # Handle missing data

    x = np.arange(len(shapes1))
    width = 0.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Adjust the bar positions based on the number of functions
    for i, (function, times) in enumerate(times_dict.items()):
        ax.bar(x + (i - len(functions) / 2) * width, times, width, label=function)

    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Benchmark: SOME VERY CATCHY TITLE')

    # create format strings for the x-axis labels
    x_labels = [shape2shape_str(shape) for shape in zip(shapes1, shapes2)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":

    # paths
    data_path = "benchmark_results.csv"
    plot_path = data_path.replace(".csv", ".png")

    # plot
    plot_einsum_str_time()