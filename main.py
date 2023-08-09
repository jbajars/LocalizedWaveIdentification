"""
Code for numerical simulations of nonlinear localized waves
in a one-dimensional crystal lattice model 
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from param_val import comp_param
from functions.energy_E import comp_energy_E
from functions.force_onsite import comp_force_onsite
from functions.force_interaction import comp_force_interaction

# Plotting properties
import matplotlib
matplotlib.rc("font", size=22) 
matplotlib.rc("axes", titlesize=22)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

print("Start of the computation!")

# Start time
start = time.time()

# Parameter values
# Number of particles
Np = 64
# Simulation time
Tsim = 200
# Dictionary <param>
param = comp_param(Np, Tsim)

# Iinitial conditions: two breathers' collision
q = param["xp"]
p = np.zeros((param["Np"], 1))
# Stationary
gamma = 0.43
p[Np//4-2] = -1*gamma
p[Np//4-1] = 2*gamma
p[Np//4] = -2*gamma
p[Np//4+1] = 1*gamma
# Moving
gamma = -0.6
p[3*Np//4-1] = -1*gamma
p[3*Np//4] = 2*gamma
p[3*Np//4+1] = -1*gamma

# Save simulation data: particle energy density values
E = np.zeros((param["Np"], param["Nsteps"]+1))
E[:,0] = comp_energy_E(q, p, param["eps"], param["Np"]).T

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
    # Save data
    E[:, n+1] = comp_energy_E(q, p, param["eps"], param["Np"]).T

# Plot particle energy density function in time
fig, ax = plt.subplots(figsize=(9, 6.5))
x, y = np.meshgrid(param["xp"]+1, param["time"])
plt.contourf(x, y, E.T+param["eps"], 100, cmap="viridis")
plt.xlabel(r"$n$")
plt.ylabel(r"$t$")
plt.axis([1, param["Np"], 0, param["Tfinal"]])
plt.title(r"Particle energy density function")
plt.colorbar()
plt.show()

# End time
end = time.time()
            
# Total time taken
print(f"Runtime of the program was {(end - start)/60:.4f} min.")
