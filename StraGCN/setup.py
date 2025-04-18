from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils import cpp_extension
import os
import sys
os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"

setup(
    name='GCN_ST',
    ext_modules=[
        CUDAExtension(
            name='GCN_ST', 
            sources=[   
                'GNN_strassen.cpp', 
                'strassen.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)