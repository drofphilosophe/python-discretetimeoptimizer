###########################
## FiniteDynamicSolver/_fds_solver.py
##
## Methods that implement the FDS solver
##
############################
import cython
import numpy
import pandas
from datetime import datetime as dt

#Find the optimal control for each value of the state
#And write these to the value function
def _findAllOptimalControls(self,t) :
    """Find optimal controls for each state in time period t conditional on a value function at t+1
    :params int t: The time period in which to compute optimal controls
    """
    #Compute the constraint mask
    self.get_constraint_mask(t)

    #Compute payoffs in this period
    self.payoff(t)

    #Compute state transitions
    self.state_transition(t)

    #Compute candidate values of the period t value function for each combination of state and control
    #This is the current period payoff plus the discounted t+1 value function for the new state
    self.vfc = self.profit_array + self.discount * self.VF[self.state_transition_idx,t+1]

    #Obtain the optimal control for each value of the state
    self.optimalControls[:,t] = numpy.argmax(self.vfc, axis=1)

    #Save the value function for each choice of optimal control
    idx = tuple([range(0,self.NS),self.optimalControls[:,t]])

    self.VF[...,t] = self.vfc[idx]

def solve(self) :
    """Solve for the optimal policy using backward induction
    """
    #Loop through time periods starting with period T and working backwards to 0
    for t in range(self.T-1,-1,-1) :
        #Extract a column vector of the exogenous variables
        start_time = dt.now()
        e = self.exogenous[t,:]

        ##SET EXOGENEOUS PARAMETERS
        self.P = e[self.eidx["P"]]

        self._findAllOptimalControls(t)
        end_time = dt.now()
        if self.verbosity > 0 :
            print("Solving t=",t,"in",end_time-start_time)


#After solving everything you have value functions and the optimal control
#But maybe you want to extract the optimal path over time given an inital state.
def get_optimal_path(self,sval_init) :
    """Compute the optimal path given an inital state
    :param numpy.array sval_init: An array of initial state values
    :returns: A pandas data frame outlining the optimal path
    """
    #Create an array to hold the optimal values of c and the evolution of the state
    #You'll compute payoffs later
    cPath = numpy.empty((self.T,self.KC),dtype=self.D_FLOAT)
    sPath = numpy.empty((self.T,self.KS),dtype=self.D_FLOAT)
    pvPath = numpy.empty((self.T,2),dtype=self.D_FLOAT)
    tPath = numpy.array(range(0,self.T),dtype=self.D_INT).reshape(self.T,1)

    #Create a list of column names for the output data frame
    colnames = ["period"] + self.stateDetails["name"] + self.controlDetails["name"] + self.exogenousDetails["name"] + ["payoff","valueFunction"]
    #copy the initial state
    sval = sval_init
    #look up the state indexes that point to the state values
    sidx = self.stateV2I(sval, flatten=True)

    #Iterate through time periods
    for t in range(0,self.T) :
        #Look up the optimal control for the state index plus time vector.
        #This returns a raveled scalar pointing to an entry in the control space
        cidx = self.optimalControls[ tuple(numpy.append(sidx,t)) ]

        #Unravel ciu to get the indexes of the optimal control in the control space
        #ci = numpy.unravel_index(ciu,self.controlShape)

        #Find the values of the control variables that correspond to the indexes in ci
        cval = self.controlI2V(cidx,unflatten=True)

        #Look up the exogenous variables
        e = self.exogenous[t]

        #Update the constraint mask
        self.get_constraint_mask(t)

        #Loop up the payoff
        self.payoff(t)

        p = self.profit_array[sidx,cidx]

        cPath[t,:] = cval
        sPath[t,:] = sval
        pvPath[t,0] = p
        pvPath[t,1] = self.VF[ tuple(numpy.append(sidx,t))]

        #Update the state
        self.state_transition(t)
        sidx = self.state_transition_idx[sidx,cidx]

        #Compute the new state values
        sval = self.stateI2V(sidx, unflatten=True)
        #si = self.stateV2I(s, flatten=False)


    df = pandas.DataFrame(
        data = numpy.hstack( (tPath,sPath,cPath,self.exogenous,pvPath) ),
        columns = colnames
        )
    return df
