from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Build import build_ext
import numpy
import pandas
import datetime
import os

ext_modules_finitehorizon_optimizer = Extension(
    "finitehorizon.optimizer",
    ["discretetimeoptimizer/finitehorizon/optimizer.py"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )

ext_modules_finitehorizon = Extension("finitehorizon",["discretetimeoptimizer/finitehorizon/__init__.py"])

setup(
    name = 'discretetimeoptimizer',
    ext_package="discretetimeoptimizer",
    ext_modules = cythonize([ext_modules_finitehorizon_optimizer],force=True),
    py_modules = ['discretetimeoptimizer/finitehorizon/__init__.py'],
    include_dirs=[numpy.get_include()],
    packages=["discretetimeoptimizer","discretetimeoptimizer.finitehorizon"]
)
