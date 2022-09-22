###########################
## FiniteDynamicSolver/_fds_initalize.py
##
## The python initalizer for Finite Dynamic Solver
##
############################
import cython
import numpy
import pandas

def _init_fds(self) :
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
    :param numpy.ndarray values: An object that can be cast as a 1-dimensional array of the variable in each time period
    :raises Exception: if the name is already defined as an exogeneous variable
    :raises ValueError: if the values cannot be recast as a numeric 1-D numpy.ndarray of length T
    """
    if name not in self.eidx.keys() :
        values = numpy.array(values,dtype=self.D_FLOAT).flatten()
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
    :param numpy.ndarray: An object that can be cast as a 1-dimensional array of the values in each time period
    :raises Exception: if the name is not currently defined as an exogeneous variable
    :raises ValueError: if the values cannot be recast as a numeric 1-D numpy.ndarray
    """
    if name in self.eidx.keys() :
        values = numpy.array(values,dtype=self.D_FLOAT).flatten()
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
            self.SA = numpy.array(self.stateDetails['minValue']).astype(numpy.float64)

            #Define the "slope" of each state transformer. It is the step size
            self.SB = numpy.array(self.stateDetails['delta']).astype(numpy.float64)

            #Define an array with the maximum index of each state variable.
            #We'll use this to keep state transitions on-grid
            self.SGMaxIndex = numpy.array( self.stateDetails['numPoints'] ) - 1

            #Get a list containing the size of each dimension in the state space
            self.stateShape = self.stateDetails['numPoints']

            #Create a 1-D array of unraveled state indexes
            self.stateIndexes = numpy.array(
                numpy.unravel_index(
                    numpy.arange(0, numpy.prod(self.stateDetails['numPoints'])).reshape(-1,) ,
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
                data = self.SA[numpy.newaxis,:] + numpy.multiply(self.SB[numpy.newaxis,:],self.stateIndexes),
                columns = self.stateDetails["name"]
            )

            self.dirty_state = False

    else :
        raise Exception("This is a dynamic solver. You need to define at least one state variable.")
        #I suppose I could make this run with no state variables, by why?

#Take a vector of state indexes and return a vector of values
#TODO: This should move to cython
def stateI2V(self,sidx,unflatten=False) -> numpy.ndarray :
    """Convert a vector of state indexes into a vector of state values

    :param numpy.ndarray sidx: A 1-dimensional array of state indexes
    :param boolean unflatten: If True unravel the scalar index, if False assume an array index
    :return: A 1-dimensional array of state values
    :rval: numpy.ndarray
    :raises IndexError: if a value in the state index is out of bounds in the state space
    """
    if unflatten :
            sidx = numpy.unravel_index(sidx,self.stateShape)

    return numpy.array(
        self.SA[numpy.newaxis,:] + numpy.multiply(self.SB[numpy.newaxis,:],sidx),
        dtype=self.D_FLOAT
        ).reshape(-1,)

#Take a vector of state values and return a vector of indexes
#TODO: This should move to cython
def stateV2I(self,sval,flatten=True) -> numpy.ndarray :
    """Convert a vector of state values into a vector of state indexes

    :param numpy.ndarray sval: A 1-dimensional array of state values
    :param boolean flatten: If true, ravel the index to a scalar, otherwise return an array of indexes
    :return: a 1-dimensional array of state indexes
    :rval: numpy.ndarray (float)
    """
    sidx = numpy.around(
        numpy.divide(numpy.array(sval) - self.SA,self.SB.transpose() ).astype(self.D_INT)
        )
    #It is possible members of S are off grid. Replace them with
    #0 or the maximum index
    numpy.clip(sidx,0,self.SGMaxIndex,out=sidx)
    if flatten :
            sidx = numpy.ravel_multi_index(sidx, self.stateShape)
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
            self.CA = numpy.array(self.controlDetails['minValue'],dtype=self.D_FLOAT)
            self.CB = numpy.array(self.controlDetails['delta'],dtype=self.D_FLOAT)

            #Create a numpy array with the maximum index for each control
            #We'll use this to be controls stay on-grid
            self.CGMaxIndex = numpy.array( self.controlDetails['numPoints'], dtype=self.D_INT  ) - 1
            self.controlShape = self.controlDetails['numPoints']

            #Create a 1-D array of raveled control indexes
            self.controlIndexes = numpy.array(
                numpy.unravel_index(
                    numpy.arange(0, numpy.prod(self.controlDetails['numPoints'])).reshape(-1,) ,
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
                data = self.CA + numpy.dot(self.controlIndexes,self.CB),
                columns = self.controlDetails["name"]
            )

            #Init an array to hold candidate values of the value function
            #This is a KC dimensional tensor where array indexes are control indexes
            #And the entries will be values of the value function for a given set of controls
            self.vfc = numpy.empty(self.controlShape)

            self.dirty_control = False
    else :
        raise Exception("You must define at least one control variable")

#Take a vector of control indexes and return a vector of values
#TODO: This should move to cython
def controlI2V(self,cidx,unflatten=False) -> numpy.ndarray:
    """Convert a vector of state indexes into a vector of state values

    :param numpy.ndarray cidx: A 1-dimensional array of control indexes
    :return: A 1-dimensional array of control values
    :rval: numpy.ndarray
    :raises IndexError: if a value in the state index is out of bounds in the state space
    """
    if unflatten :
            cidx = numpy.unravel_index(cidx,self.controlShape)
    return numpy.array(
        self.CA[numpy.newaxis,:] + numpy.multiply(self.CB[numpy.newaxis,:],cidx),
        dtype=self.D_FLOAT
        ).reshape(-1,)

#Take a vector of control values and return a vector of indexes
#TODO: This should move to cython
def controlV2I(self,cval,flatten=False) :
    """Convert a vector of control values into a vector of control indexes

    :param numpy.ndarray sval: A 1-dimensional array of control values
    :return: a 1-dimensional array of control indexes
    :rval: numpy.ndarray (float)
    """
    cidx = numpy.around(
        numpy.divide(numpy.array(cval) - self.CA,self.CB.transpose() ).astype(self.D_INT)
        )
    #It is possible members of S are off grid. Replace them with
    #0 or the maximum index
    numpy.clip(cidx,0,self.CGMaxIndex,out=cidx)
    if flatten :
        cidx = numpy.ravel_multi_index(cidx , self.controlShape)

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
            self.exogenous = numpy.empty( (self.T,self.KE), dtype=self.D_FLOAT )
            #Add each exogenous variable to the matrix, broadcasting as needed
            for i in range(0,self.KE) :
                #Broadcast values into T
                self.exogenous[:,i] = numpy.resize(self.exogenousDetails['values'][i],(self.T,))
        #If no exogenous variables were defined, make the matrix a vector of zeros
        else :
            self.exogenous = numpy.zeros( (self.T,1), dtype=self.D_FLOAT  )
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
    self.profit_array = numpy.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

    #Define a #state x #control array to hold single-period candidate values of the value function
    #There is one entry for each state-control combination
    self.vfc = numpy.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

    #Define a #state x #(time periods + 1) array to hold the value function in each period
    #Initalize it to zero. Add an extra time period with zeros for no future continuation value
    #The extra time period holds the continuation value. Currently it is set to zero
    self.VF = numpy.zeros(shape=(self.NS,self.T+1),dtype=self.D_FLOAT)

    #Define a #states x #(time periods) to hold the unraveld index of the optimlal control indexes
    #in each period.
    self.optimalControls = numpy.zeros( (self.NS,self.T) ,dtype=self.D_INT)


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
    self.state_transition_idx = numpy.empty(shape=(self.NS,self.NC),dtype=self.D_FLOAT)

    #Initialize a constraint mask. Each cell defalts to not violating a constraint (FALSE)
    self.constraint_mask = numpy.full(fill_value=False,shape=(self.NS,self.NC),dtype=self.D_BOOL)

    self._initValueFunction()
