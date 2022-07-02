# pyNeuroGMLM
Generalized multilinear model for dimensionality reduction of neural population spike trains.

The core of the code is a C++/CUDA library with optimized log likelihood and derivative computations. Using the GPU code requires a CUDA capable GPU.
CPU-only code using Numpy is also available and easier to work for with Python.

Code tested using:
CUDA 11.6
Python 3.9.7
pybind11 2.9.2
cmake 3.23.1

# Building the Python bindings

A basic library can be compiled using the cmake function and there are a couple Python scripts that setup the GMLM for GPUs.
There is a bit of demo code for building a GMLM and fitting it in **`gmlmExample.py`**.

The API requires **[pybind11](https://github.com/pybind/pybind11)**

To compile the library and run the example:
```console
user@DESKTOP:~/PROJECTHOME$ mkdir build
user@DESKTOP:~/PROJECTHOME/build$ cd build
user@DESKTOP:~/PROJECTHOME/build$ cmake ..
user@DESKTOP:~/PROJECTHOME/build$ make
user@DESKTOP:~/PROJECTHOME/Python$ cd ../Python
user@DESKTOP:~/PROJECTHOME/Python$ python gmlmExample.py
```

The CPU-only version still requires compiling the C++ library to make sure that the same code handles organizing the GMLM and trial structures - Sorry, this is an annoying and less flexible design choice on my part to not make everything in Python.
However, the library can be compiled without needing any cuda by passing in an option to cmake: <code>cmake -DWITH_GPU=Off ..</code>.

The library can be compiled to use single-precision data (double is default) using the <code>-DWITH_DOUBLE_PRECISION=Off</code> cmake option.
Given pybind11's limitations with templating, it was way easier to just require recompiling that supporting both simultaneously.
If there's any real demand to include both, I could add better support for both precisions.

## Citation (preprint)
```
Latimer, K. W., & Freedman, D. J. (2021). Low-dimensional encoding of decisions in parietal cortex reflects long-term training history. bioRxiv.
```
https://www.biorxiv.org/content/10.1101/2021.10.07.463576v1
