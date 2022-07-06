from skbuild import setup

setup(
    name="pyGMLMcuda",
    version="0.1",
    description="GPU code for generalized multilinear models for neural population data (with pybind11)",
    author='Kenneth W. Latimer',
    license={'file' : 'LICENSE'},
    packages=['pyNeuroGMLM'],
    package_dir={'': 'src'},
    cmake_install_dir='src/pyNeuroGMLM/',
    python_requires='>=3.7'#,
    #cmake_args=['-DWITH_GPU:BOOL=OFF', '-DWITH_DOUBLE_PRECISION:BOOL=OFF']
)