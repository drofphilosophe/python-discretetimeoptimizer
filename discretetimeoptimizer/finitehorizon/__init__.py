###########################
## FiniteDynamicSolver/__init__.py
##
## The header file for this class. It
## imports all of the method submodules
## and outlines class methods.
##
############################
import numpy
import pandas
from datetime import datetime as dt

class finitehorizon_optimizer :
    #Import all method submodules
    #They should all be named _fds_*
    from ._fds_initialize import _init_fds,addParam,getParam, \
        updateExogenousValues,addExogenous,setDiscountFactor,setT, \
        addControl,addState, initialize, stateV2I, stateI2V, controlV2I, controlI2V, \
        _initState, _initControl, _initExogeneous, _initValueFunction

    from ._fds_solver import solve, _findAllOptimalControls, get_optimal_path
    from ._fds_constraint_mask import get_constraint_mask
    from ._fds_payoff import payoff
    from ._fds_state_transition import state_transition

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


##Simple test code
if __name__ == "__main__" :
    fds = FiniteDynamicSolver()

    fds.addParam("gamma",1)  #Per-period losses to stored electricity
    fds.addParam("e",0.8)    #Round-trip charging efficiency
    fds.addParam("RMax",10)  #Maximum charge capacity
    fds.addParam("Omega",-100000)  #Penalty for constraint violations
    fds.addParam("EBar",4)   #Average Emissions Rate.
    fds.addParam("CV",5)    #The continuation value of unused reserves. We'll update it later. Zero for now

    fds.addState("R",-1,11,121) #Init the state grid for reserves (You need to have negative values to keep all possible moves on grid)
    #fds.addState("R2",-1,11,131)

    fds.addControl("Q",-1,1,21)  #Control grid for quantity charged (negative) or discharged (positive)

    fds.addExogenous("P",numpy.array([10,5,10,20]*6)) #Vector of prices

    fds.initialize()
    start_time = dt.now()
    for i in range(0,100) :
        fds.solve()
    end_time = dt.now()
    print("Elapsed Time:",end_time - start_time)

    x = fds.get_optimal_path(1)
    print(x)
