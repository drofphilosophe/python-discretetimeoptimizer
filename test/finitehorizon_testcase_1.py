import sys
import os
import numpy
import datetime
sys.path.append(os.path.join(".."))
import discretetimeoptimizer.finitehorizon as FDS

fds = FDS.finitehorizon_optimizer()

fds.addParam("gamma",1)  #Per-period losses to stored electricity
fds.addParam("e",0.8)    #Round-trip charging efficiency
fds.addParam("RMax",10)  #Maximum charge capacity
fds.addParam("Omega",-100000)  #Penalty for constraint violations
fds.addParam("EBar",4)   #Average Emissions Rate.
fds.addParam("CV",5)    #The continuation value of unused reserves. We'll update it later. Zero for now

fds.addState("R",-1,11,121) #Init the state grid for reserves (You need to have negative values to keep all possible moves on grid)

fds.addControl("Q",-1,1,21)  #Control grid for quantity charged (negative) or discharged (positive)

fds.addExogenous("P",numpy.array([10,5,10,20]*6)) #Vector of prices

fds.initialize()
start_time = datetime.datetime.now()
for i in range(0,1000) :
    fds.solve()
end_time = datetime.datetime.now()
print("Elapsed Time for 1000 iterations:",end_time - start_time)

x = fds.get_optimal_path(1)
print(x)
