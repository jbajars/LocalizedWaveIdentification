"""
Define functions to perform different visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

import matplotlib
matplotlib.rc("font", size=16)
matplotlib.rc("axes", titlesize=14)

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

# Define function to plot training dataset in 2D
def plot_train_data(data, labels, title):
    plt.plot(data[:, 0][labels == 1], data[:, 1][labels == 1], "bo", ms="2",
             label="Localized waves")
    plt.plot(data[:, 0][labels == 0], data[:, 1][labels == 0], "rs", ms="2",
             label="Nonlocalized waves")
    plt.title(title)

# Define function to plot testing dataset in 2D
def plot_test_data(data, labels, title):
    plt.plot(data[:, 0][labels == 1], data[:, 1][labels == 1], "gd", ms="2",
             label="Localized waves")
    plt.plot(data[:, 0][labels == 0], data[:, 1][labels == 0], "k+", ms="2",
             label="Nonlocalized waves")
    plt.title(title)
    
# Define function to plot data for decision boundary plots
def plot_data(data, gamma, n_br, dim_num = 2):
    plt.scatter(data[:-n_br,0], data[:-n_br,1], c="g", marker="s", s=1,
                label="Linear waves")
    plt.scatter(data[-n_br:,0], data[-n_br:,1], c=gamma, 
                cmap = plt.get_cmap("gnuplot"), marker="o", s=3)
    cbar = plt.colorbar(ticks=[0.25, 0.5, 0.75, 1])
    cbar.set_label("$\gamma$", y=1, ha="right", rotation=0)
    
# Define function to plot decision boundary
def plot_decision_boundary_contour(clf, axes, title = ""):
    x0s = np.linspace(axes[0], axes[1], 500)
    x1s = np.linspace(axes[2], axes[3], 500)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)      
    plt.contourf(x0, x1, y_pred, 1, cmap=plt.get_cmap("gist_gray"), alpha=0.1) 
    plt.contour(x0, x1, y_pred, 10, cmap=plt.get_cmap("Greys"), alpha=0.2)
    plt.contourf(x0, x1, y_decision, 0, cmap=plt.get_cmap("gist_gray"), alpha=0.01)
