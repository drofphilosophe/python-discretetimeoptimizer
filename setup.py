from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Build import build_ext
import numpy
import pandas
import datetime

ext_modules = [
    Extension(
        "discretetimeoptimizer",
        ["discretetimeoptimizer/finitehorizon/__init__.py",
            "discretetimeoptimizer/finitehorizon/fds_constraint_mask.py",
            "discretetimeoptimizer/finitehorizon/fds_initialize.py",
            "discretetimeoptimizer/finitehorizon/fds_payoff.py",
            "discretetimeoptimizer/finitehorizon/fds_solver.py",
            "discretetimeoptimizer/finitehorizon/fds_state_transition.py"
            ],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )
    ]

setup(
    name = 'discretetimeoptimizer',
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
