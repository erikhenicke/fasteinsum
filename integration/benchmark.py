"""
Runs the benchmarking tests for the einsum implementation against numpy's einsum.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from einsum_bmm import einsum

test_cases = [
    # ('bda,dc->abc', np.random.rand(384, 384, 384), np.random.rand(384, 24)),
    # ('dca,bd->abc', np.random.rand(384, 376, 384), np.random.rand(24, 384)),
    # ('dbea,ec->abcd', np.random.rand(96, 84, 96, 96), np.random.rand(96, 24)),
    # ('deca,be->abcd', np.random.rand(96, 84, 84, 96), np.random.rand(24, 84)),
    # ('ebad,ce->abcd', np.random.rand(96, 84, 96, 84), np.random.rand(24, 96)),
    # ('efbad,cf->abcde', np.random.rand(48, 36, 36, 48, 36), np.random.rand(24, 36)),
    # ('ecbfa,fd->abcde', np.random.rand(48, 36, 36, 48, 48), np.random.rand(48, 24)),
    # ('efcad,bf->abcde', np.random.rand(48, 36, 36, 48, 36), np.random.rand(24, 36)),
    # ('ea,ebcd->abcd', np.random.rand(96, 96), np.random.rand(96, 84, 84, 84)),
    # ('eb,aecd->abcd', np.random.rand(96, 84), np.random.rand(96, 96, 84, 84)),
    # ('ec,abed->abcd', np.random.rand(96, 84), np.random.rand(96, 84, 96, 84)),
    # ('acd,dbc->ab', np.random.rand(384, 376, 384), np.random.rand(384, 376, 376)),
    # ('cad,dcb->ab', np.random.rand(384, 384, 384), np.random.rand(384, 384, 376)),
    # ('acd,db->abc', np.random.rand(384, 376, 384), np.random.rand(384, 376)),
    ('ad,bdc->abc', np.random.rand(384, 376), np.random.rand(384, 376, 376)),
    ('adc,bd->abc', np.random.rand(384, 376, 376), np.random.rand(384, 376)),
    ('adc,db->abc', np.random.rand(384, 384, 376), np.random.rand(384, 376)),
    ('adec,ebd->abc', np.random.rand(96, 84, 96, 84), np.random.rand(96, 84, 84)),
    ('aebf,dfce->abcd', np.random.rand(96, 84, 84, 84), np.random.rand(96, 84, 84, 84)),
    # ('ac,cb->ab', np.random.rand(7248, 7248), np.random.rand(7248, 7240)),
]


bmm_einsum_fast_times = []
bmm_einsum_slow_times = []
numpy_einsum_times = []
einsum_strings = []


def einsum_fast(eq, *operands):
    return einsum(eq, *operands, useNaive=False)


def einsum_slow(eq, *operands):
    return einsum(eq, *operands, useNaive=True)


def benchmark(func, eq, *operands, expected_res=None, num_runs=10):
    times = []
    for _ in range(num_runs):
        start = time.time()
        res = func(eq, *operands)
        end = time.time()
        times.append(end - start)
        if expected_res is not None:
            assert np.allclose(res, expected_res)
    return np.mean(times)

if __name__ == '__main__':
    print("Running benchmark tests for einsum_bmm.py...")

    # Run benchmark tests
    for i, (eq, *operands) in enumerate(test_cases):
        print(f"Running test {i + 1}/{len(test_cases)}: {eq}")
        einsum_strings.append(eq)
        # res = np.einsum(eq, *operands)
        res = None
        bmm_einsum_fast_times.append(benchmark(einsum_fast, eq, *operands, expected_res=res))
        bmm_einsum_slow_times.append(benchmark(einsum_slow, eq, *operands, expected_res=res))
        # numpy_einsum_times.append(benchmark(np.einsum, eq, *operands))

    x = np.arange(len(test_cases))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    # rects1 = ax.bar(x - width, bmm_einsum_fast_times, width, label='einsum_bmm')
    # rects2 = ax.bar(x, bmm_einsum_slow_times, width, label='einsum_bmm (naive)')
    # rects3 = ax.bar(x + width, numpy_einsum_times, width, label='numpy.einsum')

    rects1 = ax.bar(x - width/2, bmm_einsum_fast_times, width, label='einsum_bmm')
    rects2 = ax.bar(x + width/2, bmm_einsum_slow_times, width, label='einsum_bmm (naive)')

    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Benchmark: einsum_bmm vs naive implementation')
    ax.set_xticks(x)
    ax.set_xticklabels(einsum_strings, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results3.png", dpi=300, bbox_inches="tight")