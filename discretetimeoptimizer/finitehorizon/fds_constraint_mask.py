###########################
## FiniteDynamicSolver/ _fds_constraint_mask.py
##
## Define the constraint mask constuctor for the finite dynamic Solver
###########################
import cython
import numpy
def get_constraint_mask(self,t) :
    """Define a constraint mask matrix
    :param int t: Time period for the constraint mask
    """
    #This method sets up a masking matrix (self.constraint_mask) that flags constraint violations
    #It is a #state x #control logical array set to true if a state/control combination violates a constraint

    #Check to see if the matrix is up-to-date. If it is, do nothing.
    #Payoffs are time-invariant, so we conly need to compute them once
    if self.last_constraint_mask_t == t :
        pass
    else :
        ###########################
        ## Constraint Masks
        ###########################
        ##Reserve constraints. These require knowledge of reserves in the next period. Compute it
        reserve_value = numpy.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)
        #Set each cell equal to the quantity produced (from the control)
        reserve_value[:,0:self.NC] = self.controlValues[self.cidx["Q"]]*numpy.where(self.controlValues[self.cidx["Q"]] > 0,self.param["e"],1)

        reserve_value[0:self.NS,:] = self.stateValues[0:self.NS,self.sidx["R"],numpy.newaxis]*self.param["gamma"] - reserve_value

        ###########
        ## Combine the constraint_mask with every constraint using logical ORs
        ###########
        #Reset the constraint mas
        self.constraint_mask[:,:] = False
        #Reserves cannot be less than zero
        numpy.logical_or(self.constraint_mask, reserve_value < 0, out=self.constraint_mask)
        #Reserves cannot be more than RMax
        numpy.logical_or(self.constraint_mask, reserve_value > self.param["RMax"], out=self.constraint_mask)

        self.last_constraint_mask_t = t
