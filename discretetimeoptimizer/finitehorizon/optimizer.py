###########################
## FiniteDynamicSolver/optimizer.py
##
## Implement a discrete time optimizer.
##
############################
#import cython
import numpy as np
import pandas
from datetime import datetime as dt
import jax.numpy as jnp

class optimizer :

    #Define data types. Chaning these may have performance implications
    #On some systems. I've selected them for maximum compatibility and flexibility
    D_FLOAT = jnp.float32
    D_INT = jnp.uint32
    D_BOOL = jnp.bool_

    #If you have static methods you should
    #declare them with
    # xxx = staticmethod(xxx)

    ########################################
    ## Class initalizer
    ########################################
    def __init__(self) :
        #############################
        ### Define our internal data structures
        #############################

        #State variables
        self.stateDetails = {
            'name' : [] ,
            'minValue' : [],
            'maxValue' : [],
            'numPoints' : [],
            'delta' : []
            }

        #Control Variables
        self.controlDetails = {
            'name' : [] ,
            'minValue' : [],
            'maxValue' : [],
            'numPoints' : [],
            'delta' : []
            }

        #Exogeneous variables
        self.exogenousDetails = {
            'name' : [],
            'values' : []
            }

        #Model parameters
        self.param = {}

        #Dictionaries that map from var names to column numbers
        self.sidx = {}  #State name -> column number
        self.cidx = {}  #Control name -> column number
        self.eidx = {}  #Exogeneous name -> column number

        self.KS = 0     #Number of state variables
        self.KC = 0     #Number of control variables
        self.KE = 0     #Number of exogeneous variables
        self.T = None   #Number of time periods
        self.discount = 1 #Discount factor

        self.verbosity = 0 #Output verbosity

        #Flags to determine if internal data structures are dirty
        self.dirty_state = True
        self.dirty_control = True
        self.dirty_exogeneous = True

        #Some flags to determine if data are dirty
        self.last_state_transition_t = None
        self.last_payoff_t = None
        self.last_constraint_mask_t = None


        self.c_solver = None


    ########################################
    ########################################
    ## Getters and Setters
    ########################################
    ########################################
    #################################
    ## Methods for setting up the optimization problem
    #################################
    #Add a state variable
    def addState(self,name,minValue,maxValue,numPoints) :
        """Add a state variable

        :param str name: Name of the state variable
        :param float minValue: Smallest possible value of the state variable
        :param float maxValue: Largest possible value of the state variable
        :param int numPoints: Number of grid points in the state space
        """
        if name not in self.sidx.keys() :
            self.sidx.update({name : self.KS})
            self.KS+=1
            self.stateDetails['name'].append(name)
            self.stateDetails['minValue'].append(minValue)
            self.stateDetails['maxValue'].append(maxValue)
            self.stateDetails['numPoints'].append(numPoints)
            self.stateDetails['delta'].append( (maxValue - minValue)/(numPoints-1) )
            self.dirty_state = True
        else :
            raise Exception("Duplicate state variable name " + name)



    #Add a control variable
    def addControl(self,name,minValue,maxValue,numPoints) :
        """Add a control variable

        :param str name: Name of the control variable
        :param float minValue: Smallest possible value of the control variable
        :param float maxValue: Largest possible value of the control variable
        :param int numPoints: Number of grid points in the control space
        """
        if name not in self.cidx.keys() :
            self.cidx.update({name : self.KC})
            self.KC+=1
            self.controlDetails['name'].append(name)
            self.controlDetails['minValue'].append(minValue)
            self.controlDetails['maxValue'].append(maxValue)
            self.controlDetails['numPoints'].append(numPoints)
            self.controlDetails['delta'].append( (maxValue - minValue)/(numPoints-1) )
            self.dirty_control = True
        else :
            raise Exception("Duplicate controle variable name " + name)






    #######################
    ## Number of periods
    #######################
    def setT(self,T) :
        """Set the number of time periods

        :param int T: Number of time periods
        """
        try :
            T = int(T)
        except ValueError :
            raise ValueError("T must be an integer")
        if T <= 0 :
            raise ValueError("T must be a positive integer")

        self.T = T

    #######################
    ## The discount factor
    #######################
    def setDiscountFactor(self,d) :
        """Set the discount factor

        :param float d: A discount factor on the (0,1] interval
        """
        try :
            d = float(d)
        except ValueError :
            raise ValueError("d must be a floating point number")

        if d <= 0 or d > 1 :
            raise ValueError("d must be on the (0,1] interval")

        self.discount = d



    #######################
    ## Exogenous variables
    #######################
    #Add an exogenous variable.
    #When we init the values will be broadcasted to be as long as the number of time periods
    def addExogenous(self,name,values) :
        """Add a time-varying exogeneous variable

        :param str name: Name of the exogeneous variable
        :param jnp.ndarray values: An object that can be cast as a 1-dimensional array of the variable in each time period
        :raises Exception: if the name is already defined as an exogeneous variable
        :raises ValueError: if the values cannot be recast as a numeric 1-D jnp.ndarray of length T
        """
        if name not in self.eidx.keys() :
            values = jnp.array(values,dtype=self.D_FLOAT).flatten()
            if self.T is None :
                self.setT(len(values))
            elif len(values) != self.T :
                raise ValueError("Exogeneous value vector should have length equal to the number of time periods.")

            self.eidx.update({name : self.KE})
            self.exogenousDetails['name'].append(name)
            self.exogenousDetails['values'].append(values)
            self.dirty_exogeneous = True
            self.KE+=1
        else :
            raise Exception("You previously defined an exogeneous variable named" + str(name))

    def updateExogenousValues(self,name,values) :
        """Update the time series of an exogeneous value

        :param str name: Name of an existing exogeneous variable
        :param jnp.ndarray: An object that can be cast as a 1-dimensional array of the values in each time period
        :raises Exception: if the name is not currently defined as an exogeneous variable
        :raises ValueError: if the values cannot be recast as a numeric 1-D jnp.ndarray
        """
        if name in self.eidx.keys() :
            values = jnp.array(values,dtype=self.D_FLOAT).flatten()
            if len(values) != self.T :
                raise ValueError("Exogeneous value vector should have length equal to the number of time periods.")

            idx = self.eidx[name]
            self.exogenousDetails['values'][idx] = values
            self.dirty_exogeneous = True
        else :
            raise Exception("Unknown exogeneous variable name. You must add an exogeneous variable before updating its values")

    ##################
    ## Define parameters
    ##################
    def addParam(self,name,value) :
        """Add a model parameter

        :param str name: Name of the parameter
        :param float value: Value of the parameter
        :raises ValueError: if value cannot be recast as a float
        """
        if name in self.param :
            self.param[name] = float(value)
        else :
            self.param.update( {name : float(value)} )

    def getParam(self,name) :
        """Return the value of a model parameter

        :param str name: Name of the parameter
        :return: Value of the parameter
        :rtype: float
        :raises KeyError: if parameter name has not been defined
        """
        return self.param[name]



    #Initialize a Python version of the state space.
    #This provides a facility to map from python indices back to values of the state
    #The state space for each variable evenly divides the state into numPoints components
    #This makes computing state values much faster since, given state index si and
    #vectors self.SA and self.SB we can compute the value of the state as
    # s = self.SA + si*self.SB
    def _initState(self) :
        """Initalize a python version of the state space

        :raises Exception: if you attempt to initialize the state space without defining any state variables
        """
        #Check that we have defined some state variables
        if self.KS > 0 :
            #Only rebuild the state space if it's dirty
            if self.dirty_state == True :
                #Define the "intercept" of the state transformer. It is the minimum vaulue of each state variable
                self.SA = jnp.array(self.stateDetails['minValue']).astype(self.D_FLOAT)

                #Define the "slope" of each state transformer. It is the step size
                self.SB = jnp.array(self.stateDetails['delta']).astype(self.D_FLOAT)

                #Define an array with the maximum index of each state variable.
                #We'll use this to keep state transitions on-grid
                self.SGMaxIndex = jnp.array( self.stateDetails['numPoints'] ).astype(self.D_INT) - 1

                #Get a list containing the size of each dimension in the state space
                self.stateShape = jnp.array(self.stateDetails['numPoints'])

                #Create a 1-D array of unraveled state indexes
                self.stateIndexes = jnp.array(
                    jnp.unravel_index(
                        jnp.arange(0, jnp.prod(self.stateShape)).reshape(-1,) ,
                        self.stateShape
                        ),
                    dtype=self.D_INT
                    ).transpose()

                #Create a 1-D array of state values
                self.stateValues = (self.SA + self.SB*self.stateIndexes).astype(self.D_FLOAT)

                #Count the number of states
                self.NS = self.stateValues.shape[0]

                #Now compute a data frame where each row is a vector of state values
                #We'll name the columns to make it easier to write functions that use these values
                #THIS IS A DATAFRAME WHERE EACH ROW IS A STATE VALUE VECTOR
                self.SFlatDF = pandas.DataFrame(
                    data = self.SA[jnp.newaxis,:] + jnp.multiply(self.SB[jnp.newaxis,:],self.stateIndexes),
                    columns = self.stateDetails["name"]
                )

                self.dirty_state = False

        else :
            raise Exception("This is a dynamic solver. You need to define at least one state variable.")
            #I suppose I could make this run with no state variables, by why?

    #Take a vector of state indexes and return a vector of values
    #TODO: This should move to cython
    def stateI2V(self,sidx,unflatten=False) -> jnp.ndarray :
        """Convert a vector of state indexes into a vector of state values

        :param jnp.ndarray sidx: A 1-dimensional array of state indexes
        :param boolean unflatten: If True unravel the scalar index, if False assume an array index
        :return: A 1-dimensional array of state values
        :rval: jnp.ndarray
        :raises IndexError: if a value in the state index is out of bounds in the state space
        """
        if unflatten :
                sidx = jnp.unravel_index(sidx,self.stateShape)

        return jnp.array(
            self.SA[np.newaxis,:] + np.multiply(self.SB[jnp.newaxis,:],sidx),
            dtype=self.D_FLOAT
            ).reshape(-1,)

    #Take a vector of state values and return a vector of indexes
    #TODO: This should move to cython
    def stateV2I(self,sval,flatten=True) -> jnp.ndarray :
        """Convert a vector of state values into a vector of state indexes

        :param jnp.ndarray sval: A 1-dimensional array of state values
        :param boolean flatten: If true, ravel the index to a scalar, otherwise return an array of indexes
        :return: a 1-dimensional array of state indexes
        :rval: jnp.ndarray (float)
        """
        sidx = jnp.around(
            jnp.divide(jnp.array(sval) - self.SA,self.SB.transpose() ).astype(self.D_INT)
            )
        #It is possible members of S are off grid. Replace them with
        #0 or the maximum index
        sidx = jnp.clip(sidx,0,self.SGMaxIndex)
        if flatten :
                sidx = jnp.ravel_multi_index(sidx, self.stateShape)
        return sidx




    ###############################
    #### Control Variables
    ###############################



    #Init the control space.
    #Again degine a slope and intercept to map control indexes to control values
    def _initControl(self) :
        """Initialize a python version of the control space

        :raises Exception: if you attempt to initialize the control space without defining any control variables
        """

        if self.KC > 0 :
            #Only rebuild the control space if its dirty
            if self.dirty_control == True :
                self.CA = jnp.array(self.controlDetails['minValue'],dtype=self.D_FLOAT)
                self.CB = jnp.array(self.controlDetails['delta'],dtype=self.D_FLOAT)

                #Create a jnp array with the maximum index for each control
                #We'll use this to be controls stay on-grid
                self.CGMaxIndex = jnp.array( self.controlDetails['numPoints'], dtype=self.D_INT  ) - 1
                self.controlShape = jnp.array(self.controlDetails['numPoints'])

                #Create a 1-D array of raveled control indexes
                self.controlIndexes = jnp.array(
                    jnp.unravel_index(
                        jnp.arange(0, jnp.prod(self.controlShape)).reshape(-1,) ,
                        self.controlShape
                        ),
                    dtype=self.D_INT
                    ).transpose()

                #Create a vector of control values
                self.controlValues = (self.CA + self.CB*self.controlIndexes).transpose()
                #Count the number number of control states
                self.NC = self.controlValues.shape[1]

                #Now compute a matrix where each row is a vector of state values
                #We'll name the columns to make it easier to write functions that use these values
                #THIS IS A MATRIX WHERE EACH ROW IS A STATE VALUE VECTOR
                self.CFlatDF = pandas.DataFrame(
                    data = self.CA + jnp.dot(self.controlIndexes,self.CB),
                    columns = self.controlDetails["name"]
                )

                #Init an array to hold candidate values of the value function
                #This is a KC dimensional tensor where array indexes are control indexes
                #And the entries will be values of the value function for a given set of controls
                self.vfc = jnp.empty(self.controlShape)

                self.dirty_control = False
        else :
            raise Exception("You must define at least one control variable")

    #Take a vector of control indexes and return a vector of values
    #TODO: This should move to cython
    def controlI2V(self,cidx,unflatten=False) -> jnp.ndarray:
        """Convert a vector of state indexes into a vector of state values

        :param jnp.ndarray cidx: A 1-dimensional array of control indexes
        :return: A 1-dimensional array of control values
        :rval: jnp.ndarray
        :raises IndexError: if a value in the state index is out of bounds in the state space
        """
        if unflatten :
                cidx = jnp.unravel_index(cidx,self.controlShape)
        return jnp.array(
            self.CA[jnp.newaxis,:] + np.multiply(self.CB[jnp.newaxis,:],cidx),
            dtype=self.D_FLOAT
            ).reshape(-1,)

    #Take a vector of control values and return a vector of indexes
    #TODO: This should move to cython
    def controlV2I(self,cval,flatten=False) :
        """Convert a vector of control values into a vector of control indexes

        :param jnp.ndarray sval: A 1-dimensional array of control values
        :return: a 1-dimensional array of control indexes
        :rval: jnp.ndarray (float)
        """
        cidx = jnp.around(
            jnp.divide(jnp.array(cval) - self.CA,self.CB.transpose() ).astype(self.D_INT)
            )
        #It is possible members of S are off grid. Replace them with
        #0 or the maximum index
        jnp.clip(cidx,0,self.CGMaxIndex,out=cidx)
        if flatten :
            cidx = jnp.ravel_multi_index(cidx , self.controlShape)

        return cidx





    ##############################
    ## Exogeneous variables
    ##############################
    def _initExogeneous(self) :
        """Initialize python versions of the exogeneous variables

        """
        #Only rebuild the exogeneous variables if they're dirty
        if self.dirty_exogeneous == True :
            #If we defined exogenous variables, copy them to a matrix
            if self.KE > 0 :
                #Init a matrix to hold exogenous values
                #Jax arrays are immutable. We'll construct a numpy array and then copy it
                local_exogenous = np.empty( (self.T,self.KE), dtype=self.D_FLOAT )
                #Add each exogenous variable to the matrix, broadcasting as needed
                for i in range(0,self.KE) :
                    #Broadcast values into T
                    local_exogenous[:,i] = np.resize(self.exogenousDetails['values'][i],(self.T,))
                
                self.exogenous = jnp.array(local_exogenous)
            #If no exogenous variables were defined, make the matrix a vector of zeros
            else :
                self.exogenous = jnp.zeros( (self.T,1), dtype=self.D_FLOAT  )
                #Set self.KE to one to simulate a single exogenous variable
                self.KE = 1

            self.dirty_exogeneous = True

        ####################
        ## Value Function
        ####################
        #The value function is a 2-D state x time array
        #That maps a state to value of the optimal path in t+1 forward

    #First init the value function
    def _initValueFunction(self) :
        """
        Initialize data structres needed to compute the value function.
        This should be run after initalizing states, controls, and exogeneous variables
        """
        #Define a #state x #control array to hold single period profits for each possible
        #State-Control combination
        self.profit_array = jnp.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

        #Define a #state x #control array to hold single-period candidate values of the value function
        #There is one entry for each state-control combination
        self.vfc = np.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

        #Define a #state x #(time periods + 1) array to hold the value function in each period
        #Initalize it to zero. Add an extra time period with zeros for no future continuation value
        #The extra time period holds the continuation value. Currently it is set to zero
        self.VF = np.zeros(shape=(self.NS,self.T+1),dtype=self.D_FLOAT)

        #Define a #states x #(time periods) to hold the unraveld index of the optimlal control indexes
        #in each period.
        self.optimalControls = np.zeros( (self.NS,self.T) ,dtype=self.D_INT)


    ########################
    ## The Grand Initalizer
    ########################
    def initialize(self) :
        """Build all data structures in preparation for running the solver

        """
        self._initState()
        self._initControl()
        self._initExogeneous()

        #Initialize a state transition index array. it is #state x #control array where each entry is a new raveled state index
        self.state_transition_idx = jnp.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

        #Initialize a constraint mask. Each cell defalts to not violating a constraint (FALSE)
        self.constraint_mask = jnp.full(fill_value=False,shape=(self.NS,self.NC),dtype=self.D_BOOL)

        self._initValueFunction()


    ##################################################
    ##################################################
    ### Solver
    ##################################################
    ##################################################
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
        self.vfc = np.array(self.profit_array + self.discount * self.VF[self.state_transition_idx,t+1])

        #Obtain the optimal control for each value of the state
        self.optimalControls[:,t] = jnp.argmax(self.vfc, axis=1)

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
        :param jnp.array sval_init: An array of initial state values
        :returns: A pandas data frame outlining the optimal path
        """
        #Create an array to hold the optimal values of c and the evolution of the state
        #You'll compute payoffs later
        cPath = np.empty((self.T,self.KC),dtype=self.D_FLOAT)
        sPath = np.empty((self.T,self.KS),dtype=self.D_FLOAT)
        pvPath = np.empty((self.T,2),dtype=self.D_FLOAT)
        tPath = np.array(range(0,self.T),dtype=self.D_INT).reshape(self.T,1)

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
            cidx = self.optimalControls[ tuple(jnp.append(sidx,t)) ]

            #Unravel ciu to get the indexes of the optimal control in the control space
            #ci = jnp.unravel_index(ciu,self.controlShape)

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
            pvPath[t,1] = self.VF[ tuple(jnp.append(sidx,t))]

            #Update the state
            self.state_transition(t)
            sidx = self.state_transition_idx[sidx,cidx]

            #Compute the new state values
            sval = self.stateI2V(sidx, unflatten=True)
            #si = self.stateV2I(s, flatten=False)


        df = pandas.DataFrame(
            data = np.hstack( (tPath,sPath,cPath,self.exogenous,pvPath) ),
            columns = colnames
            )
        return df





    ###########################################
    ###########################################
    ## Payoff function
    ###########################################
    ###########################################
    def payoff(self,t) :
        """Compute a payoff matrix
        :param int t: The time period in which to compute payoffs
        """
        #Check to see if the payoff matrix is up-to-date. If it is, do nothing.
        if t == self.last_payoff_t :
            pass
        else :
            local_profit_array = np.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

            #P*Q profit doesn't depend on the state. Update all of the entries
            local_profit_array[:,0:self.NC] = self.exogenous[t,self.eidx["P"]]*self.controlValues[self.cidx["Q"]]

            #Choosing zero quantitiy results in zero profits
            local_profit_array[ : , self.controlValues[self.cidx["Q"]] == 0 ] = 0

            local_profit_array[self.constraint_mask] = self.param["Omega"]

            #Add a continuation value in the final period
            #Here I'll make it net reserves times a constant
            if t == self.T - 1 :
                #Create a matrix of continuation values for each state/control
                local_continuation_value = np.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)
                local_continuation_value[:,0:self.NC] = self.controlValues[self.cidx["Q"]]
                local_continuation_value[0:self.NS,:] = (self.stateValues[0:self.NS,self.sidx["R"]].reshape((self.NS,1)) - local_continuation_value[0:self.NS,:])*self.param["CV"]
                local_profit_array = self.profit_array + local_continuation_value
                self.continuation_value = jnp.array(local_continuation_value)
            else :
                self.continuation_value = jnp.zeros(shape=(self.NS,self.NC),dtype=self.D_FLOAT)
            #return self.profit_array

            
            self.profit_array = jnp.array(local_profit_array)

            self.last_payoff_t = t


    ############################################
    ############################################
    ## State Transition function
    ############################################
    ############################################
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
            new_state_vals = [ self.stateValues[:,idx,jnp.newaxis] * jnp.ones_like(self.controlValues) for idx in range(0,self.KS) ]

            #Compute new reserves as previous reserves times gamme
            new_state_vals[self.sidx["R"]] = self.stateValues[:,self.sidx["R"],jnp.newaxis]*self.param["gamma"] - \
                                  jnp.where(self.controlValues[jnp.newaxis,self.cidx["Q"]], self.param["e"], 1) * self.controlValues[jnp.newaxis,self.cidx["Q"]]

            #Quantize each new_state_vals to the grid
            for s in range(0,self.KS) :
                new_state_vals[s] = jnp.around((new_state_vals[s] - self.SA[s])/self.SB[s]).astype(jnp.int64)

            self.state_transition_idx = jnp.ravel_multi_index(new_state_vals,dims=self.stateShape,mode='clip')

            self.last_state_transition_t = t



    ##############################################
    ##############################################
    ## Constraint Mask
    ##############################################
    ##############################################
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
            reserve_value = np.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)
            #Set each cell equal to the quantity produced (from the control)
            reserve_value[:,0:self.NC] = self.controlValues[self.cidx["Q"]]*np.where(self.controlValues[self.cidx["Q"]] > 0,self.param["e"],1)

            reserve_value[0:self.NS,:] = self.stateValues[0:self.NS,self.sidx["R"],np.newaxis]*self.param["gamma"] - reserve_value

            ###########
            ## Combine the constraint_mask with every constraint using logical ORs
            ###########
            #Reset the constraint mask
            #self.constraint_mask = jnp.full_like(self.constraint_mask,False)
            #self.constraint_mask[:,:] = False
            #Reserves cannot be less than zero
            #jnp.logical_or(self.constraint_mask, reserve_value < 0, out=self.constraint_mask)
            #Reserves cannot be more than RMax
            #jnp.logical_or(self.constraint_mask, reserve_value > self.param["RMax"], out=self.constraint_mask)

            self.constraint_mask = jnp.logical_or(reserve_value < 0,reserve_value > self.param["RMax"])
            self.last_constraint_mask_t = t
