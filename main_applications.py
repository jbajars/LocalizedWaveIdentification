"""
Code for nonlinear localized waves identification in numerical simulations of 
one-dimensional crystal lattice model 
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib as jl
from param_val import comp_param
from functions.energy_E import comp_energy_E
from functions.force_onsite import comp_force_onsite
from functions.force_interaction import comp_force_interaction
from functions.density import comp_density

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

# Load classifier
mth = "lle"
kernel = "linear"
data_title = "X"
# Number of simulations used
Nsim = 1000
# Number of neighboring particles
Nd = 8
clf = jl.load("saved_classifiers/clf_" + mth + "_" + kernel + "_" + data_title + 
              "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".pkl")
dim_red = jl.load("saved_classifiers/" + mth + "_" + data_title + 
                  "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".pkl")
scaler = jl.load("saved_classifiers/" + mth + "_scaler_" + data_title + 
                 "_Nsim" + str(Nsim) + "Nd" + str(Nd) +".pkl") 

# Initial conditions: two breathers' collision
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

# Save simulation data
# Positions
Q = np.zeros((param["Np"], param["Nsteps"]+1))
Q[:,0] = q.T
# Displacements
W = np.zeros((param["Np"], param["Nsteps"]+1))
# Compute particle displacement values taking into account 
# periodic boundary conditions
w = q - param["xp"]
if q[0] > param["Np"]/2:
    w[0] -= param["Np"]
if q[-1] < param["Np"]/2:
    w[0] += param["Np"]
W[:, 0] = w.T
# Momenta
P = np.zeros((param["Np"], param["Nsteps"]+1))
P[:,0] = p.T
# Energy density values
E = np.zeros((param["Np"], param["Nsteps"]+1))
E[:,0] = comp_energy_E(q, p, param["eps"], param["Np"]).T
# Localization density values
rho = np.zeros((param["Np"], param["Nsteps"]+1))
e = np.reshape(E[:, 0].T, q.shape)
rho[:, 0] = comp_density(w, p, E[:, 0].T, Nd, clf, dim_red, scaler, data_title)

# Time integration with the Verlet method
for n in range(param["Nsteps"]):
    # Update positions
    qq = q + param["tau"]/2*p
    # Compute forces
    G = comp_force_onsite(qq)
    F = comp_force_interaction(qq, param["eps"], param["Np"])
    # Update momenta
    p = p + param["tau"]*(G+F)
    # Update positions
    q = qq + param["tau"]/2*p

    # Save data
    Q[:, n+1] = q.T
    P[:, n+1] = p.T
    E[:, n+1] = comp_energy_E(q, p, param["eps"], param["Np"]).T
    e = np.reshape(E[:, n+1].T, q.shape)
    # Compute particle displacement values
    w = q - param["xp"]
    if q[0] > param["Np"]/2:
        w[0] -= param["Np"]
    if q[-1] < param["Np"]/2:
        w[0] += param["Np"]
    # Save particle displacement values
    W[:, n+1] = w.T
    # Compute localization density function values at time t=(n+1)*tau
    rho[:, n+1] = comp_density(w, p, e, Nd, clf, dim_red, scaler, data_title)

# Save simulation data into folder saved_applications_data
np.savetxt("saved_applications_data/Q_applications_" + mth + "_" + 
           kernel + "_" + data_title + "_Nsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", Q, delimiter=",")
np.savetxt("saved_applications_data/W_applications_" + mth + "_" + 
           kernel + "_" + data_title + "_Nsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", W, delimiter=",")
np.savetxt("saved_applications_data/P_applications_" + mth + "_" + 
           kernel + "_" + data_title + "_Nsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", P, delimiter=",")
np.savetxt("saved_applications_data/E_applications_" + mth + "_" + 
           kernel + "_" + data_title + "_Nsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", E, delimiter=",")
np.savetxt("saved_applications_data/rho_applications_" + mth + "_" + 
           kernel + "_" + data_title + "_Nsim" + str(Nsim) + "Nd" + str(Nd) + 
           ".csv", rho, delimiter=",")

# Plot particle energy density function in time
fig, ax = plt.subplots(figsize=(9, 6.5))
x, y = np.meshgrid(param["xp"]+1, param["time"])
im = plt.contourf(x, y, E.T+param["eps"], 100, cmap="viridis")
plt.xlabel(r"$n$")
plt.ylabel(r"$t$")
plt.axis([1, param["Np"], 0, param["Tfinal"]])
plt.title(r"Particle energy density function")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)
plt.colorbar(im, cax=cax)
# Save figure
plt.savefig("figures/particle_energy_density" + 
            "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".png", 
            dpi=300, bbox_inches="tight")
plt.show()

# Plot normalized localization density function in time
fig, ax = plt.subplots(figsize=(9, 6.5))
x, y = np.meshgrid(param["xp"]+1, param["time"])
im = plt.contourf(x, y, rho.T/Nd, Nd, cmap="plasma")
plt.xlabel(r"$n$")
plt.ylabel(r"$t$")
plt.axis([1, param["Np"], 0, param["Tfinal"]])
plt.title(r"Normalized localization density function")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)
plt.colorbar(im, cax=cax, ticks=np.round(np.arange(0,2,1/Nd),2))
# Save figure
plt.savefig("figures/normalized_localization_density" + 
            "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".png", 
            dpi=300, bbox_inches="tight")
plt.show()

# End time
end = time.time()
            
# Total time taken
print(f"Runtime of the program was {(end - start)/60:.4f} min.")
