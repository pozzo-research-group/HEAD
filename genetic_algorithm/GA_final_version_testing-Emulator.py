#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as mms 
from GA_functions import fitness, select_parents, crossover, mutation, GA_algorithm, GA_algorithm_unnormalized, conc_to_spectra, perform_iteration, set_seed #normalize_and_pca
from tree_search_functions import zeroth_iteration, nth_iteration, plot_fitness, plot_spectra, plot_DLS
from SAXS_model import model, model2
from spectroscopy import obtain_spectra


# ## Define Emulator

# Define the emulator as a function.

# In[2]:


def perform_simulations(conc_array, wavelength_1, wavelength_2):
    for i in range(conc_array.shape[0]):
        #Emulator 1: SAXS curve
        spectra_1_row = model(wavelength_1, 10*conc_array[i,0], 3*conc_array[i,1]).reshape(1,-1)
        #Emulator 2: SAXS curve
        spectra_2_row = model2(wavelength_1, 10*conc_array[i,0], 3*conc_array[i,1]).reshape(1,-1) 
        if i == 0:
            spectra_array_1 = spectra_1_row
            spectra_array_2 = spectra_2_row
        else:
            spectra_array_1 = np.vstack((spectra_array_1, spectra_1_row))
            spectra_array_2 = np.vstack((spectra_array_2, spectra_2_row))
    return spectra_array_1, spectra_array_2


# ### Initialize first iteration of Input parameters and Output parameters

# In[3]:


np.random.seed(2)
conc_array = np.random.dirichlet((1, 1), 30)
q1 = np.linspace(1e-1, 1, 81)
q2 = np.linspace(1e-1, 1, 81)
spectra_array_1, spectra_array_2 = perform_simulations(conc_array, q1, q2) 


# ### Initialize Targets

# These are the spectra that are the targets of the optimization. The algorithm will try to find the best input parameters to create a sample that has a spectra closest to the two targets.

# In[4]:


desired_spectra_1 = model(q1, 5,2).reshape(1,-1) 
desired_spectra_2 = model2(q2, 2,.5).reshape(1,-1) 


# ## Zeroth Iteration

# These cells initialize the experiment by plotting the fitness score of the initial sample's spectra/ scattering curve compared to the target spectra/ scattering curve. It will also plot all the spectra/ scattering curve.

# In[5]:


desired_spectra_1 = desired_spectra_1.reshape(-1,1)
desired_spectra_2 = desired_spectra_2.reshape(-1,1)
loaded_dict = zeroth_iteration(conc_array = conc_array, spectra_array_1 = spectra_array_1, desired_spectra_1 = desired_spectra_1, spectra_array_2 = spectra_array_2, desired_spectra_2 = desired_spectra_2)


# In[6]:


plot_spectra(loaded_dict, wavelength_1 = q1, wavelength_2 = q2, savefig = False)


# ## Nth Iteration 

# Run the cells starting from here all the way to the end of the notebook to perform an additional iteration. A plot of the maximum and median fitness over the iterations will be created and plots of all the sample's spectra/scattering curve compared to the target spectra/scattering curve will be generated.

# In[7]:


Iterations = 25 #sample size for GA 
Moves_ahead = 3 #moves ahead that are calculated 
GA_iterations = 8 #times per move that the GA is used 
n_samples = 30 #sample size
seed = 2
loaded_dict = nth_iteration(loaded_dict, seed = seed, Iterations = Iterations, Moves_ahead = Moves_ahead, GA_iterations = GA_iterations, n_samples = n_samples)


# In[8]:


loaded_dict['next_gen_conc'][-1,:] = loaded_dict['best_conc_array'][:-1]
loaded_dict['next_gen_conc'][-2,:] = loaded_dict['best_candidate_array'][-1, 0:-1]


# In[9]:


spectra_array_1, spectra_array_2 = perform_simulations(loaded_dict['next_gen_conc'], q1, q2)
loaded_dict['conc_array_actual'] = np.vstack((loaded_dict['conc_array_actual'],loaded_dict['next_gen_conc']))
loaded_dict['spectra_array_actual_1'] = np.vstack((loaded_dict['spectra_array_actual_1'], spectra_array_1))
loaded_dict['spectra_array_actual_2'] = np.vstack((loaded_dict['spectra_array_actual_2'], spectra_array_2))
loaded_dict['spectra_array_1'] = spectra_array_1
loaded_dict['spectra_array_2'] = spectra_array_2


# In[10]:


median_fitness_list, max_fitness_list, iteration, best_candidate_array = plot_fitness(loaded_dict, savefig = False)
loaded_dict['best_candidate_array'] = best_candidate_array
loaded_dict['median_fitness_list'] = median_fitness_list
loaded_dict['max_fitness_list'] = max_fitness_list


# In[11]:


plot_spectra(loaded_dict, wavelength_1 = q1, wavelength_2 = q2, savefig = False)


# In[ ]:





# In[ ]:




