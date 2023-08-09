"""
Code to compute total energy (Hamiltonian) H
"""
import numpy as np

# Define function comp_energy
def comp_energy(q, p, eps, N):
    # Add kinetic and on-site potential energies
    H = p.T@p/2 + np.sum(1 - np.cos(2*np.pi*q))
    # Compute interaction potential energies
    for n in range(N):
        if n == N-1:
            r = abs(q[0] + N - q[n])
        else:
            r = abs(q[n+1] - q[n])
        r = (1/r)**6
        # Add interaction potential energy to H
        H = H + eps*(r**2 - 2*r)
    return H
