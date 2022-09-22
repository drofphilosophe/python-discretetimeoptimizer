# python-discretetimeoptimizer

## Extensible dynamic, discrete time optimizers in Python

This package will provide a suite of discrete time optimizer for arbitrary optimal control problems. They are designed to compute policy functions and then simulate optimal paths conditional on an intial state. This package has the following overaching goals:
* *Extensibility* - Solve arbitrary optimal control problems with minimial modification of the code. The long-run goal is that problem parameters could be loaded from a human-readable configuration file and translated into Python or machine code. Currently, defining state, control, and exogeneous variables and their discretizations is straighforward and handled by class methods. Defining constriants, payoffs, and state transitions requires overriding class methods with new versions.
* *Speed* - The solver relies primarily on `numpy` for high-performance matrix calculations. The solver code has been optimized to find solutions quickly, sometimes at the cost of interpreitability and readability of the solver code. The solver methods are sufficiently generic that one could setup and solve optimztion problems without taking the time to understand how it works. 

### Finite-Horizon Optimization (`finitehorizon_optimizer`)

This package currently provides an extensible non-stochastic, discrete-time, finite horizon optimal control solver in `discretetimeoptimzer.finitehorizon_optimizer`. It will solve for the policy function of arbitary dynamic optimization problems using backward induction. Problems may include multiple states, controls, and exogeneous variables. 

As the package is currently implemented, much of the configutation is manual. To use the package, you should create a subclass that overrides the following methods:
* `payoff(self,t)` - This method returns nothing, but has the side effect of setting the value of `self.profit_array` to the payoff for each state and control combination.
* `state_transtion(self,t)` - This method returns nothing but has the side effect of setting the value of `self.state_transition_idx` to the raveled index of the `t+1` state for each combination of time `t` states and controls
* `constraint_mask(self,t)` - This method returns nothing but has the side effect of setting the value of `self.constraint_mask` to `True` for every state-control combination that violates a model constraint and `False` otherwise

After that, one should instantiate the class, define state, control, and exogeneous variables and parameters then call `solve()` to solve for the policy function. Finally `get_optimal_path(s)` where `s` is an array representing the initail state, will compute optimal paths. 
