###########################
## FiniteDynamicSolver/ _fds_payoff.py
##
## Define the payoff function for the finite dynamic Solver
###########################
import numpy

def payoff(self,t) :
    """Compute a payoff matrix
    :param int t: The time period in which to compute payoffs
    """
    #Check to see if the payoff matrix is up-to-date. If it is, do nothing.
    if t == self.last_payoff_t :
        pass
    else :
        #P*Q profit doesn't depend on the state. Update all of the entries
        self.profit_array[:,0:self.NC] = self.exogenous[t,self.eidx["P"]]*self.controlValues[self.cidx["Q"]]

        #Choosing zero quantitiy results in zero profits
        self.profit_array[ : , self.controlValues[self.cidx["Q"]] == 0 ] = 0

        self.profit_array[self.constraint_mask] = self.param["Omega"]

        #Add a continuation value in the final period
        #Here I'll make it net reserves times a constant
        if t == self.T - 1 :
            #Create a matrix of continuation values for each state/control
            self.continuation_value = numpy.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)
            self.continuation_value[:,0:self.NC] = self.controlValues[self.cidx["Q"]]
            self.continuation_value[0:self.NS,:] = (self.stateValues[0:self.NS,self.sidx["R"]].reshape((self.NS,1)) - self.continuation_value[0:self.NS,:])*self.param["CV"]
            self.profit_array = self.profit_array + self.continuation_value
        #return self.profit_array

        self.last_payoff_t = t
