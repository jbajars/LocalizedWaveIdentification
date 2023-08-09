"""
Define parameter values in dictionary <param>
"""
import numpy as np

# Define function comp_param 
def comp_param(Np, Tsim):
    # Values of epsilon, final time, number of particles, and time step 
    param = {
        "eps": 0.05,
        "Tfinal": Tsim,
        "Np": Np,
        "tau": 0.01
    }
    # Number of time steps
    param["Nsteps"] = int(param["Tfinal"]/param["tau"])
    # Time grid points
    param["time"] = np.linspace(0, param["Tfinal"],
                                param["Nsteps"]+1).reshape(param["Nsteps"]+1,1)
    # Particle equilibrium positions
    param["xp"] = np.arange(0, param["Np"], 1).reshape(param["Np"],1)
    return param
