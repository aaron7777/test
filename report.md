# Fast Python on CUDA-capable GPUs

## Abstract

Though Python has many advantages such as its flexibility and simple expressive syntax, it is considered as poor performance in terms of massive data processes and computation. To accelarate the computation speed of Python, we came up with the idea of using CUDA-capable GPUs. In our final project, we focused on NumbaPro and tried several ways to improve performance like using 'vectorize', CUDA Host API and writing CUDA directly in Python. Up to 700 times of speedup is achieved in the final tests.


## Introduction

Numba is a Numpy-aware optimizing compiler for Python. Numba supports the just-in-time compilation from original Python code to machine code wih the LLVM compiler infrastructure, leading to improved performance.

![image](https://github.com/aaron7777/pic/raw/master/1.jpg)

NumbaPro is the enhanced version of Numba. NumbaPro compiler targets multi-core CPU and GPUs directly from simple Python syntax, which enables easily move vectorized NumPy functions to the GPU and has multiple CUDA device support.

Except this, NumbaPro provides a Python interface to CUDA cuBLAS (dense linear algebra), cuFFT (Fast Fourier Transform), and cuRAND (random number generation)libraries. And its CUDA Python API provides explicit control over data transfer and CUDA streams.

With NumbaPro, we tried THREE ways to improve the python computing ability. They are using 'vectorize' to automatically accelerate, binding to cuRAND, cuFFT host API for operation on Numpy arrays and writing CUDA directly in Python. We found samples to test how much 'vectorize' and host API could help and then implemented two examples (bubble sort and blowfish encrption) by writing CUDA in Python.


## Sample Test
Testing GPU: GeForce GTX 870M

### Monte Carlo Option Pricer via Numpy

The following is a general Numpy example to implement Monte Carlo Option Pricer and its performance, here is part of the code:

```python

import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
import math

def step_numpy(dt, prices, c0, c1, noises):
    return prices * np.exp(c0 * dt + c1 * noises)

def mc_numpy(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in xrange(1, paths.shape[1]):   # for each time step
        prices = paths[:, j - 1]          # last prices
        # gaussian noises for simulation
        noises = np.random.normal(0., 1., prices.size)
        # simulate
        paths[:, j] = step_numpy(dt, prices, c0, c1, noises)
```

<img src="https://raw.githubusercontent.com/shaowei-su/CSC453/master/NumbaproTests/1.png" width="400px" height="188px">


### Acceleration by CUDA Vectorize

CUDA vectorize produce a universal-function-like object, which is close analog but not fully compatible with a regular Numpy u-func. The CUDA vectorized u-func adds support for passing intra-device arrays (already on the GPU device) to reduce traffic over the PCI-express bus. 

To use it just simply change target to 'gpu', here is code snippet:

```python

import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
import math
from numbapro import vectorize

@vectorize(['f8(f8, f8, f8, f8, f8)'], target='gpu')
def step_gpuvec(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def mc_gpuvec(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in xrange(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = step_gpuvec(prices, dt, c0, c1, noises)

```

<img src="https://raw.githubusercontent.com/shaowei-su/CSC453/master/NumbaproTests/4.png" width="400px" height="188px">

As can be seen from the result, there is a 25% speed up without changing large part of original code.

While due to the memory transfer overheads(from CPU memory to GPU global memory), the performance here is still not excellent.

### CUDA JIT(Just in Time) 

To deal with the problem mentioned above, a cuRAND random number generator with CUDA JIT feature compile is used to reduce memory transfer overheads.

NumbaPro’s CUDA JIT is able to compile CUDA Python functions at run time and realize explicit control over data transfers and CUDA streams. The sample code goes like this:

```python
import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
import math
from numbapro import cuda, jit
from numbapro.cudalib import curand

@jit('void(double[:], double[:], double, double, double, double[:])', target='gpu')
def step_cuda(last, paths, dt, c0, c1, normdist):
    i = cuda.grid(1)
    if i >= paths.shape[0]:
        return
    noise = normdist[i]
    paths[i] = last[i] * math.exp(c0 * dt + c1 * noise)

def mc_cuda(paths, dt, interest, volatility):
    n = paths.shape[0]

    blksz = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    gridsz = int(math.ceil(float(n) / blksz))

    # instantiate a CUDA stream for queueing async CUDA cmds
    stream = cuda.stream()
    # instantiate a cuRAND PRNG
    prng = curand.PRNG(curand.PRNG.MRG32K3A)

    # Allocate device side array
    d_normdist = cuda.device_array(n, dtype=np.double, stream=stream)
    
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    # configure the kernel
    # similar to CUDA-C: step_cuda<<<gridsz, blksz, 0, stream>>>
    step_cfg = step_cuda[gridsz, blksz, stream]
    
    # transfer the initial prices
    d_last = cuda.to_device(paths[:, 0], stream=stream)
    for j in range(1, paths.shape[1]):
        # call cuRAND to populate d_normdist with gaussian noises
        prng.normal(d_normdist, mean=0, sigma=1)
        # setup memory for new prices
        # device_array_like is like empty_like for GPU
        d_paths = cuda.device_array_like(paths[:, j], stream=stream)
        # invoke step kernel asynchronously
        step_cfg(d_last, d_paths, dt, c0, c1, d_normdist)
        # transfer memory back to the host
        d_paths.copy_to_host(paths[:, j], stream=stream)
        d_last = d_paths
    # wait for all GPU work to complete
    stream.synchronize()

```

<img src="https://raw.githubusercontent.com/shaowei-su/CSC453/master/NumbaproTests/5.png" width="400px" height="188px">

With the help of JIT compile and cuRAND library, we finally get a 18x speed up.


## Write CUDA in Python

## Future Work

## Conclusion

## Reference