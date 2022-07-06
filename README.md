# pyNeuroGMLM
Generalized multilinear model for dimensionality reduction of neural population spike trains.

The core of the code is a C++/CUDA library with optimized log likelihood and derivative computations. Using the GPU code requires a CUDA capable GPU.
CPU-only code using Numpy is also available and easier to work for with Python.

Code tested using:
CUDA 11.6
Python 3.9.7
pybind11 2.9.2
cmake 3.23.1

There are a couple Python scripts that demonstrate the very basic setup the GMLM for GPUs and on CPU in <code>examples/</code>.

# Installing with pip

```console
pip install git+https://github.com/latimerkw/pyNeuroGMLM/
```

The basic library will be compiled using cmake.
The install uses **[pybind11](https://github.com/pybind/pybind11)** to compile the CUDA/C++ library and hooks it up to Python.

The CPU-only version still requires compiling the C++ library to make sure that the same code handles organizing the GMLM and trial structures - Sorry, this is an annoying and less flexible design choice on my part to not make everything in Python.
If no CUDA install is detected, the GPU code is disabled by default.
Additionally, the library can be compiled without using Cuda by setting an environment variable before the install command: 

```console
export GMLM_WITH_GPU=OFF
pip install -v git+https://github.com/latimerkw/pyNeuroGMLM/ 
```

For some reason, passing in an option directly into pip caused trouble, and replacing it with an environment variable was a quick and dirty fix.
WINDOWS USERS: use <code>set GMLM_WITH_GPU=OFF</code> instead of <code>export</code>.

The library can be compiled to use single-precision data (double is default) using the <code>GMLM_WITH_DOUBLE_PRECISION=Off</code> option as (or in additon to) the GPU option above.
Given pybind11's limitations with templating, it was way easier to just require recompiling that supporting both simultaneously.
If there's any real demand to include both, I could add better support for both precisions.


## Citation (preprint)
```
Latimer, K. W., & Freedman, D. J. (2021). Low-dimensional encoding of decisions in parietal cortex reflects long-term training history. bioRxiv.
```
https://www.biorxiv.org/content/10.1101/2021.10.07.463576v1


**[MATLAB version for the DMC task](https://github.com/latimerkw/GMLM_DMC)** 
