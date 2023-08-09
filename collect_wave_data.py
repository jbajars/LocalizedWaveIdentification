"""
Script file to collect local wave data from numerical simulations
of one-dimensional crystal lattice model 
"""
import numpy as np
import time
from param_val import comp_param
from functions.energy_E import comp_energy_E
from functions.force_onsite import comp_force_onsite
from functions.force_interaction import comp_force_interaction

# Function to set initial conditions
def comp_p_init(Np, gamma, n_init_core):
    m = np.zeros((Np, 1))
    m[n_init_core] = -1*gamma
    m[n_init_core+1] = 2*gamma
    m[n_init_core+2] = -2*gamma
    m[n_init_core+3] = 1*gamma
    return m

# Funtion to obtain indexes  
def comp_n_edges(n_init_core, Nd):
    if Nd <= 4:
        n_init = n_init_core
        n_final = n_init_core + Nd - 1
    else:
        Nd_res = Nd - 4
        if Nd_res%2 == 0:
            n_init = n_init_core - Nd_res/2
            n_final = n_init_core + 3 + Nd_res/2
        else:
            n_init = n_init_core - (Nd_res+1)/2
            n_final = n_init_core + 3 + (Nd_res-1)/2
    return int(n_init), int(n_final)

print("Start of the computation!")

# Start time
start = time.time()

# Parameter values
# Gamma limit values
gamma_init = 0.15
gamma_final = 1.2
# Number of simulations to collect data from
Nsim = 1000
# Number of neighboring particles from which to obtain data
Nd = 8
# Bretaher\phonon wave proportion
br_prop = 0.75
dp = 0.01
dq = 0.01
# Number of particles
Np = 64
# Simulation time
Tsim = 3
# Dictionary <param>
param = comp_param(Np, Tsim)
# Gamma values
n_br = int(br_prop*Nsim)
gamma = np.zeros((Nsim, 1))
# Random seed
np.random.seed(0)
gamma[-n_br:] = (gamma_final - gamma_init)*np.random.rand(n_br, 1) + gamma_init

# Initial conditions
n_init_core = int(0.4*param["Np"])
n_init, n_final = comp_n_edges(n_init_core, Nd)
q_init = param["xp"]
# To add random initial conditions 
np.random.seed(1)
P = np.random.uniform(-dp, dp, (Nsim-n_br, param["Np"])).T
np.random.seed(2)
Q = np.random.uniform(-dq, dq, (Nsim-n_br, param["Np"])).T

# For saving data: displacements, momenta, and particle energy density values
data = np.zeros((Nsim, Nd*3))

# Loop over the number of simulations Nsim
for sim in range(Nsim):
    print(sim+1)
    
    # Diferent initial conditions to obtain different wave data
    if gamma[sim] == 0:
        # Initial conditions for linear wave data
        p = P[:, sim].reshape(param["Np"], 1)
        q = param["xp"] + Q[:, sim].reshape(param["Np"], 1)
    else:
        # Initial conditions for nonlinear wave data
        p = comp_p_init(param["Np"], gamma[sim], n_init_core)
        q = param["xp"]
        
    # Time integration with the Verlet method
    for n in range(param["Nsteps"]):
        # Update positions
        qq = q + param["tau"]/2*p
        # Compute forces
        G = comp_force_onsite(qq)
        F = comp_force_interaction(qq, param["eps"], param["Np"])
        # Update momenta
        p = p + param["tau"]*(G + F)
        # Update positions
        q = qq + param["tau"]/2*p
        
    # Save local wave data in matrix data
    data[sim,:Nd] = q[n_init:n_final+1].T - q_init[n_init:n_final+1].T
    data[sim,Nd:-Nd] = p[n_init:n_final+1].T
    data[sim,-Nd:] = comp_energy_E(q, p, param["eps"], 
                                   param["Np"])[n_init:n_final+1].T

# Save data in csv and txt files
np.savetxt("saved_sim_data/dataNsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", data, delimiter=",")
np.savetxt("saved_sim_data/gammaNsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", gamma[-n_br:], delimiter=",")
np.savetxt("saved_sim_data/br_propNsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".txt", np.array([br_prop]))
print("The data was saved!")

# End time
end = time.time()
            
# Total time taken
print(f"Runtime of the program was {(end - start)/60:.4f} min.")
