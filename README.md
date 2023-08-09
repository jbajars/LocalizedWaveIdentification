# LocalizedWaveIdentification

Python code for identification of localized energy and nonlinear waves in numerical simulations of one-dimensional crystal lattice models. Using the code and developed methodology, please cite Bajārs, J., Kozirevs, F.: *Data-driven intrinsic localized mode detection and classification in one-dimensional crystal lattice model*. Physics Letters A **436**, 128071 (2022), [DOI: 10.1016/j.physleta.2022.128071](https://doi.org/10.1016/j.physleta.2022.128071).

File `LocalizedWaveIdentification_PureCode_WithoutData.zip` contains pure code without precomputed data and images.

<p float="left">
  <img src="figures/LLE_X_linear_Nsim1000Nd6.png" width="30%" /> &nbsp;    
  <img src="figures/particle_energy_density_Nsim1000Nd6.png" width="33%" /> &nbsp;   
  <img src="figures/normalized_localization_density_Nsim1000Nd6.png" width="33%" /> 
</p>

This research has been financially supported by the specific support objective activity 1.1.1.2. “Post-doctoral Research Aid” of the Republic of Latvia (Project No. 1.1.1.2/VIAA/4/20/617 “Data-Driven Nonlinear Wave Modelling”), funded by the European Regional Development Fund (project id. N. 1.1.1.2/16/I/001).

#### Instructions to run the code
- To perform a numerical simulation of the lattice dynamics, run the file `main.py`.
- All the parameter values are saved in the dictionary *param* and set in the file `param_val.py`.
- All functions used during the numerical simulations, applications and visualizations are defined in the folder `functions`.
- All images are saved in the folder `figures`.
- To collect different wave data from numerical simulations, run the file `collect_wave_data.py`.
- All collected wave data from numerical simulations is saved in the folder `saved_sim_data`.
- To obtain classification algorithms with two dimensionality rediction algorithms run the files `classification_PCA.py` and `classification_LLE`.
- All trained classification algorithms are saved in the folder `saved_classifiers`.
- To test and obtain precision and recall scores, run the file `precision_recall_mean.py`.
- To identify nonlinear localized waves in numerical simulations using the built classification algorithms and sliding window approac, run the file `main_aaplications.py`, where all the data is saved in the folder `saved_applications_data`.
