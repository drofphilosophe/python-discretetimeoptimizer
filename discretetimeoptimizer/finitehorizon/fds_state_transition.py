###########################
## FiniteDynamicSolver/ _fds_state_transition.py
##
## Define the state transition constuctor for the finite dynamic Solver
###########################
import cython
import numpy
def state_transition(self,t) :
    """Compute a state transition matrix
    :param int t: The time period in which to compute state transitions
    """
    #Check to see if the payoff matrix is up-to-date. If it is, do nothing.
    #Payoffs are time-invariant, so we conly need to compute them once
    if self.last_state_transition_t is not None :
        pass
    else :
        #Create a list of #state x #control arrays, defaulting to the current value of each state
        new_state_vals = [ self.stateValues[:,idx,numpy.newaxis] * numpy.ones_like(self.controlValues) for idx in range(0,self.KS) ]

        #Compute new reserves as previous reserves times gamme
        new_state_vals[self.sidx["R"]] = self.stateValues[:,self.sidx["R"],numpy.newaxis]*self.param["gamma"] - \
                              numpy.where(self.controlValues[numpy.newaxis,self.cidx["Q"]], self.param["e"], 1) * self.controlValues[numpy.newaxis,self.cidx["Q"]]

        #Quantize each new_state_vals to the grid
        for s in range(0,self.KS) :
            new_state_vals[s] = numpy.around((new_state_vals[s] - self.SA[s])/self.SB[s]).astype(numpy.int64)

        self.state_transition_idx = numpy.ravel_multi_index(new_state_vals,dims=self.stateShape,mode='clip')

        self.last_state_transition_t = t
