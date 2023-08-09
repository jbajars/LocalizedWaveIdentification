"""
Code to compute localization density function
"""
import numpy as np

# Define function comp_density
def comp_density(w, p, e, Nd, clf, dim_red, scaler, data_title):
    
    # Number of particles 
    Np = np.shape(w)[0]
    
    # Localization density function 
    rho = np.zeros((Np,))
    
    # Data type: W, WP, or X
    if data_title == "W":
        data = np.empty((1, Nd*1))
    elif data_title == "WP":
        data = np.empty((1, Nd*2))
    else:
        data = np.empty((1, Nd*3))
    
    # Find local data
    w_prep = np.empty((Nd,1))
    p_prep = np.empty((Nd,1))
    e_prep = np.empty((Nd,1))
    
    # Loop over all particles
    for i in range(Np):
        
        # First particle index of the window
        n_init = i
        
        # Last particle index of the window
        n_final = n_init + Nd - 1
        
        # Index vector
        idx = np.linspace(n_init, n_final, Nd).astype(int)
        
        # Apply mod function to the vector idx
        idx = idx % Np
        
        # Local data at indexes idx
        w_prep = w[idx]
        p_prep = p[idx]
        e_prep = e[idx]
        
        # Classification data: W, WP, or X
        data[:, :Nd] = w_prep.T
        if data_title == "WP":
            data[:, Nd:2*Nd] = p_prep.T
        if data_title == "X":
            data[0, Nd:2*Nd] = p_prep.T
            data[0, 2*Nd:] = e_prep.T
        
        # Perform classification
        data_test = dim_red.transform(data)
        data_test_s = scaler.transform(data_test)
        pred = clf.predict(data_test_s)
        
        # Add one to rho if classified as a nonlinear wave location
        if pred[0] == 1:
            rho[idx] += 1        
    return rho
