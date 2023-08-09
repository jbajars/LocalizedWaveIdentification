"""
Code to compute particle interaction forces 
"""
import numpy as np

# Define function comp_force_interaction
def comp_force_interaction(q, eps, Np):
    # Save computed interaction forces in vector F for each particle
    F = np.zeros((Np, 1))
    for n in range(Np):
        # Compute distances
        if n == 0:
            Q1 = q[n] + Np - q[Np-1]
        else:
            Q1 = q[n] - q[n-1]
        if n == Np-1:
            Q2 = q[0] + Np - q[n]
        else:
            Q2 = q[n+1] - q[n]
        r1 = abs(Q1)
        r2 = abs(Q2)
        # Compute forces between two neighboring particles
        F1 = 12*eps/r1**2*((1/r1)**12 - (1/r1)**6)*Q1
        F2 = 12*eps/r2**2*((1/r2)**12 - (1/r2)**6)*Q2
        # Save forces
        F[n] = F1 - F2
    return F
