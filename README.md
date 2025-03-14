# Einsum with batch matrix multiplication
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

## Running Tests
To verify the installation, run the test suite:

```sh
pytest tests -v
```

## Support
If any support is needed, we are there to help. Reach out to us under
- erik.henicke@uni-jena.de or
- sonja.weitzing@uni-jena.de