"""
Code for dimensionality reduction with PCA and training SVCs  
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from functions.visualization import plot_data, plot_decision_boundary_contour
from functions.visualization import plot_train_data, plot_test_data

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

# Preparing data
# Number of simulations used
Nsim = 1000
# Number of neighboring particles
Nd = 4
# Load data from folder saved-sim_data
X_init = np.genfromtxt("saved_sim_data/dataNsim" + str(Nsim) + "Nd" + str(Nd) + 
                       ".csv", delimiter = ",")
gamma = np.genfromtxt("saved_sim_data/gammaNsim" + str(Nsim) + "Nd" + str(Nd) + 
                      ".csv", delimiter = ",")
br_prop = np.genfromtxt("saved_sim_data/br_propNsim" + str(Nsim) + "Nd" + str(Nd) + 
                        ".txt")
Nd = int(np.shape(X_init)[1]/3)
n_br = int(br_prop*Nsim)
y = np.zeros((Nsim,))
y[-n_br:] = np.ones((n_br,))

# Loop over three different datasets: X, W, and WP
for i in range(3):
        
    # Choosing the dataset
    if i == 0:
        X = np.copy(X_init)
        data_title = "X"
    elif i == 1:
        X = X_init[:,:Nd]
        data_title = "W"
    else:
        X = X_init[:, :2*Nd]
        data_title = "WP"

    # Split dataset X into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        shuffle=True)
    
    # Dimensionality reduction of training dataset with PCA
    pca = PCA(n_components=2, svd_solver="full")   
    X_train_pca = pca.fit_transform(X_train)
    print(pca.explained_variance_ratio_)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print(cumsum[-1])
    scaler = StandardScaler()
    X_train_pca_s = scaler.fit_transform(X_train_pca)
    
    # Save PCA projection and scaler
    jl.dump(pca, "saved_classifiers/pca_" + data_title + 
            "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".pkl")
    jl.dump(scaler, "saved_classifiers/pca_scaler_" + data_title + 
            "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".pkl")
   
    # Training data visualization
    fig, ax = plt.subplots(figsize=(9, 6.5))
    plot_train_data(X_train_pca_s, y_train, data_title + r"$_{train}$" + " PCA")
    plt.grid("on")
    plt.xlabel(r"PC1")
    plt.ylabel(r"PC2")

    # Model training; choose the kernel based on the dataset, i.e.,
    # X - linear, W and WP - nonlinear 
    if i==0:
        grid_search = GridSearchCV(
            SVC(random_state=2021, kernel="linear"),
            param_grid = {"C": [0.0001, 0.01, 0.1, 1, 10, 20, 40, 60, 80]},
            cv = 5)
    else:
        grid_search = GridSearchCV(
            SVC(random_state=2021, kernel="rbf"),
            param_grid = {"C": [0.0001, 0.01, 0.1, 1, 10, 20, 40, 60, 80]},
            cv = 5)
    grid_search.fit(X_train_pca_s, y_train)
    print("======================================================")
    print("Optimal model" + " (" + data_title + "):",
          grid_search.best_estimator_)
    clf = grid_search.best_estimator_
    clf.fit(X_train_pca_s, y_train)
    print(clf._gamma)
    
    # Save classifier
    jl.dump(clf, "saved_classifiers/clf_pca_" + clf.kernel + "_" + data_title + 
            "_Nsim" + str(Nsim) + "Nd" + str(Nd) + ".pkl") 
    
    # Performance measure
    cv_score = cross_val_score(clf, X_train_pca_s, y_train, cv=5, scoring="accuracy")
    y_train_pred = cross_val_predict(clf, X_train_pca_s, y_train, cv=5)
    prec = precision_score(y_train, y_train_pred)
    rec = recall_score(y_train, y_train_pred)
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("\nModel performance measure (" + data_title + ")\n")
    print("Cross-Validation score (accuracy):", cv_score)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Confusion matrix:")
    print(conf_matrix)

    # Perform testing on the testing dataset
    X_test_pca = pca.transform(X_test)
    X_test_pca_s = scaler.transform(X_test_pca)
    y_test_pred = clf.predict(X_test_pca_s)

    # Testing data visualization
    plot_test_data(X_test_pca_s, y_test, data_title + r"$_{train}$" + " PCA")
    plt.show()
      
    prec_final = precision_score(y_test, y_test_pred)
    rec_final = recall_score(y_test, y_test_pred)
    conf_matrix_final = confusion_matrix(y_test, y_test_pred)
    print("\nEvaluating on the test set (" + data_title + ")\n")
    print("Final precision:", prec_final)
    print("Final recall:", rec_final)
    print("Final confusion matrix:")
    print(conf_matrix_final)

    # Decision boundary plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    X_transf = pca.transform(X)
    X_transf_s = scaler.transform(X_transf)
    axes = [-5, 5, -5, 5]
    plot_data(X_transf_s, gamma, n_br)
    plot_decision_boundary_contour(clf, axes,
                                   " (" + data_title + ")" + ", LinearSVC")
    plt.xlabel(r"PC1")
    plt.ylabel(r"PC2")
    plt.legend(prop={"size": 20}, shadow=True, ncol=1)
    plt.axis([-3, 3, -2, 4])
    if i==0:
        plt.savefig("figures/PCA_X_linear" + "_Nsim" + str(Nsim) + 
                    "Nd" + str(Nd) + ".png", dpi=300, bbox_inches="tight")
    elif i==1:
        plt.savefig("figures/PCA_W_rbf" + "_Nsim" + str(Nsim) + 
                    "Nd" + str(Nd) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig("figures/PCA_WP_rbf" + "_Nsim" + str(Nsim) + 
                    "Nd" + str(Nd) + ".png", dpi=300, bbox_inches="tight")
    plt.show()
    
# End time
end = time.time()
            
# Total time taken
print(f"Runtime of the program was {(end - start)/60:.4f} min.")
    