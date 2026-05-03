# Fast Einsum with batch matrix multiplication
This repository contains an einsum python library that uses batch matrix multiplication created by,
- Sonja Weitzing and
- Erik Henicke
  in the context of the course Algorithm Engineering by Mark Blacher at Friedrich Schiller University Jena.

## Motivation
The goal of this library is to improve the performance of the einsum implementation in NumPy via batch matrix multiplications.
Generally, [tensor contractions](https://www.tensors.net/p-tutorial-1) can be mapped onto efficient GEMM (General Matrix Multiplication) implementations by transposing the transposing and reshaping the dimensions.
Without batch dimensions this mapping is described as [Transpose-Transpose-GEMM-Transpose](http://publications.rwth-aachen.de/record/755345/files/755345.pdf). In the general case of einsum operations, batch dimensions can be present making need for different kernels that perform batch matrix multiplications (BMM).
This [einsum-BMM](https://github.com/jcmgray/einsum_bmm/blob/main/einsum_bmm.py) approach was already implemented in python.
The aim of this project is to develop an efficient `C++` library for BMM which provides a python interface and build a python library arround it providing fast einsum functionallity.

## Folder structure
- `bmm` contains the `C++` library for batch matrix multiplication.
- `tests` contains the tests for the bmm library.
- `einsum_benchmark` contains the benchmark and tests for the fast einsum library.
- `fast_einsum` contains the python library that uses the `C++` library for batch matrix multiplication.
- `results` contains the results of the benchmarks.
- `plot` contains the scripts to plot the results.

## Installation
The folder `fast_einsum` contains the python einsum library that uses the `C++` library for batch matrix multiplication 
and can be distributed as a python package: `fast_einsum/dist/fast_einsum-0.1.0-py3-none-any.whl`.

Simply install the package via pip:
```bash
pip install fast_einsum-0.1.0-py3-none-any.whl
```

If you want to perform the tests, you can install the package with the test dependencies:
```bash
fast_einsum/dist/fast_einsum-0.1.0-py3-none-any.whl[test]
```
and run the tests from the `fast_einsum` directory:
```bash
pytest tests -v
```

## Support
If any support is needed, we are there to help. Reach out to us under
- erik.henicke@uni-jena.de or
- sonja.weitzing@uni-jena.de
