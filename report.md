# Fast Python on CUDA-capable GPUs

## Abstract

Though Python has many advantages such as its flexibility and simple expressive syntax, it is considered as poor performance in terms of massive data processes and computation. To accelarate the computation speed of Python, we came up with the idea of using CUDA-capable GPUs. In our final project, we focused on NumbaPro and tried several ways to improve performance like using 'vectorize', CUDA Host API and writing CUDA directly in Python. Up to 700 times of speedup is achieved in the final tests.


## Introduction

Numba is a Numpy-aware optimizing compiler for Python. Numba supports the just-in-time compilation from original Python code to machine code wih the LLVM compiler infrastructure, leading to improved performance.

![image](https://github.com/aaron7777/pic/raw/master/1.jpg)

NumbaPro is the enhanced version of Numba. NumbaPro compiler targets multi-core CPU and GPUs directly from simple Python syntax, which enables easily move vectorized NumPy functions to the GPU and has multiple CUDA device support.

Except this, NumbaPro provides a Python interface to CUDA cuBLAS (dense linear algebra), cuFFT (Fast Fourier Transform), and cuRAND (random number generation)libraries. And its CUDA Python API provides explicit control over data transfer and CUDA streams.

With NumbaPro, we tried THREE ways to speed up Python programs. They are using 'vectorize' to automatically accelerate, binding to cuRAND, cuFFT host API for operation on Numpy arrays and writing CUDA directly in Python. We found samples to test how much 'vectorize' and host API could help and then implemented two examples (bubble sort and blowfish encrption) by writing CUDA in Python.


## Sample Test

## Write CUDA in Python

## Future Work

## Conclusion

## Reference