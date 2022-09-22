###########################
## FiniteDynamicSolver/__init__.py
##
## The header file for this class. It
## imports all of the method submodules
## and outlines class methods.
##
############################
import cython
import numpy
import pandas
from datetime import datetime as dt

class finitehorizon_optimizer :
    #Import all method submodules
    #They should all be named _fds_*
    from .fds_initialize import _init_fds,addParam,getParam, \
        updateExogenousValues,addExogenous,setDiscountFactor,setT, \
        addControl,addState, initialize, stateV2I, stateI2V, controlV2I, controlI2V, \
        _initState, _initControl, _initExogeneous, _initValueFunction

    from .fds_solver import solve, _findAllOptimalControls, get_optimal_path
    from .fds_constraint_mask import get_constraint_mask
    from .fds_payoff import payoff
    from .fds_state_transition import state_transition

    #Define data types. Chaning these may have performance implications
    #On some systems. I've selected them for maximum compatibility and flexibility
    D_FLOAT = numpy.float64
    D_INT = numpy.uint64
    D_BOOL = numpy.bool

    #If you have static methods you should
    #declare them with
    # xxx = staticmethod(xxx)

    def __init__(self) :
        self._init_fds()
