from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("FiniteDynamicSolver", ["__init__.py"]),
    Extension("constraint_masl", [r"_fds_constraint_mask.py"]),
    Extension("initialize", [r"_fds_initialize.py"]),
    Extension("payoff", [r"_fds_payoff.py"]),
    Extension("solver", [r"_fds_solver.py"]),
    Extension("state_transition", [r"_fds_state_transition.py"])
]

ext_modules = [
    Extension(
        "FiniteDynamicSolver.FiniteDynamicSolver",
        ["__init__.py","_fds_constraint_mask.py","_fds_initialize.py","_fds_payoff.py","_fds_solver.py","_fds_state_transition.py"]
        ),
    ]

setup(
    name = 'FiniteDynamicSolver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()]
)
