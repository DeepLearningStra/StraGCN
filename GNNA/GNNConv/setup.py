import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"
setup(
    name='GNNAdvisor',
    ext_modules=[
        CUDAExtension(
        name='GNNAdvisor', 
        sources=[   
                    'GNNAdvisor.cpp', 
                    'GNNAdvisor_kernel.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })