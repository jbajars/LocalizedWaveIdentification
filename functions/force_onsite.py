"""
Code to compute particle on-site forces 
"""
from numpy import pi, sin

# Define function comp_force_onsite
def comp_force_onsite(q):
    # Compute on-site force for each particle 
    F = -2*pi*sin(2*pi*q)
    return F
