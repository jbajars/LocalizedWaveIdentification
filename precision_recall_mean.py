"""
Code to compute averaged precision and recall scores
of different classification algorithms
"""
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

print("Start of the computation!")

# Start time
start = time.time()

# Number of simulations used
Nsim = 1000
# Number of neighboring particles
Nd = 8
# Load data from folder saved-sim_data
X_init = np.genfromtxt("saved_sim_data/dataNsim" + str(Nsim) + "Nd" + str(Nd) + 
                       ".csv", delimiter = ",")
Nd = int(np.shape(X_init)[1]/3)
y = np.zeros((Nsim,))
br_prop = np.genfromtxt("saved_sim_data/br_propNsim" + str(Nsim) + "Nd" + str(Nd) + 
                        ".txt")
n_br = int(br_prop*Nsim)
y[-n_br:] = np.ones((n_br,))

# Averaged results over Nk data splits
Nk = 10
# Save precision and recall values
prec_train = np.empty((Nk,))
rec_train = np.empty((Nk,))
prec_test = np.empty((Nk,))
rec_test = np.empty((Nk,))
c_sum = np.empty((Nk,))

# Loop over three different datasets: X, W, and WP
for k in range(3):
    
    # Choosing the dataset
    if k == 0:
        X = np.copy(X_init)
        data_title = "X"
    elif k == 1:
        X = X_init[:,:Nd]
        data_title = "W"
    else:
        X = X_init[:,:2*Nd]
        data_title = "WP"

    # Loop over kernels and dimensionality reduction techniques
    for j in range(4):
        
        # Choose between linear and nonlinear kernels
        if j in [0, 1]:
            kernel = "linear"
        else:
            kernel = "rbf"
        
        # Choose between PCA and LLE
        if j in [0, 2]:
            dim_red = PCA(n_components = 2)
        else:
            dim_red = LocallyLinearEmbedding(n_components=2, n_neighbors=20)

        # Loop over Nk data splits
        for i in range(Nk):
            
            # Split dataset X into training and testing datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size = 0.3,
                                                                random_state=5*(i+1),
                                                                shuffle=True)
            # Compute scaler
            scaler = StandardScaler()
            # Diemensionality reduction
            X_train = dim_red.fit_transform(X_train)
            # Apply scaler
            X_train = scaler.fit_transform(X_train) 
            
            # Compute cumulative sum for PCA only
            if j in [0, 2]:
                c_sum[i] = np.cumsum(dim_red.explained_variance_ratio_)[1]
            
            # Perform grid search 
            grid_search = GridSearchCV(
                    SVC(kernel=kernel, random_state=2003),
                    param_grid = {"C": [0.0001, 0.01, 0.1, 1, 10, 20, 40, 60, 80]},
                    cv = 5,
                )
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            clf.fit(X_train, y_train)
            
            # Compure scores on training data
            y_train_pred = cross_val_predict(clf, X_train, y_train, cv=5)
            prec_train[i] = precision_score(y_train, y_train_pred)
            rec_train[i] = recall_score(y_train, y_train_pred)
            
            # Compure scores on testing data
            X_test = dim_red.transform(X_test)
            X_test = scaler.transform(X_test)
            y_test_pred = clf.predict(X_test)
            prec_test[i] = precision_score(y_test, y_test_pred)
            rec_test[i] = recall_score(y_test, y_test_pred)
            
        # Print results
        print("\n======================================================\n")
        print("Data: " + data_title)
        print("Kernel: " + kernel)
        print("Dimensionality reduction technique:", dim_red) 
        if j in [0, 2]:
            print("Cumulative sum of variance: ", "{:.4}".format(np.mean(c_sum)))
        print("Precision (validation):", "{:.4}".format(np.mean(prec_train)))
        print("Recall (validation):", "{:.4}".format(np.mean(rec_train)))
        print("Precision (testing):", "{:.4}".format(np.mean(prec_test)))
        print("Recall (testing):", "{:.4}".format(np.mean(rec_test)))

# End time
end = time.time()
            
# Total time taken
print(f"\nRuntime of the program was {(end - start)/60:.4f} min.")
