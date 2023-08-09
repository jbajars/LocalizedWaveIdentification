"""
Code to compute particle energy density function E 
"""
import numpy as np

# Define function comp_energy 
def comp_energy_E(q, p, eps, Np):
    # Save energy values in vector E 
    E = np.zeros((Np, 1))
    # Kinetic energy and on-site potential values 
    E = E + (p*p)/2 + (1 - np.cos(2*np.pi*q))
    # Add interaction potential half values to each particle
    for n in range(Np):
        if n == 0:
            r1 = abs(q[n]-q[Np-1]+Np)
            r2 = abs(q[n+1]-q[n])
        elif n == Np-1:
            r1 = abs(q[n]-q[n-1])
            r2 = abs(q[0]+Np-q[n])
        else:
            r1 = abs(q[n]-q[n-1])
            r2 = abs(q[n+1]-q[n])
        r1 = (1/r1)**6
        r2 = (1/r2)**6
        # Add interaction potential energy values
        E[n] = E[n] + eps*((r1**2 - 2*r1) + (r2**2 - 2*r2))/2
    return E
