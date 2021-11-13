import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.model_selection import train_test_split
from GA_functions import fitness, select_parents, crossover, mutation, GA_algorithm, GA_algorithm_unnormalized, conc_to_spectra, perform_iteration, set_seed #normalize_and_pca
from tree_search_functions import zeroth_iteration, nth_iteration, plot_fitness, plot_spectra, plot_DLS
from Prepare_Data_Functions import load_df, subtract_baseline, normalize_df, delete_rows, plot_all_spectra_single


class single_objective:
    def __init__(self):
        self.itr = 0
        return

    def initialize_exp(self):
        # CREATE THE FIRST ITERATION USING RANDOM SAMPLING
        np.random.seed(2)
        self.conc_array = np.random.rand(15,5)
        self.conc_array_actual = self.conc_array
        idir = '../x_values'
        os.makedirs(idir)


    def load_target(self, name, loc):
        # IMPORT TARGET SPECTRA
        df_desired = pd.read_excel(name)
        #df_desired = subtract_baseline(df_desired, 'A2')
        df_desired = normalize_df(df_desired)
        #df_desired = df_desired.drop(['A2'], axis = 1)
        #x_test = df_desired['Target'].values.reshape(-1,1)
        self.x_test = df_desired[loc].values.reshape(-1,1)
        self.x_test = mms().fit(self.x_test).transform(self.x_test).T
        self.x_test = self.x_test.reshape(1, -1)[0].reshape(-1, 1).T

    def plot_results(self):
        plot_spectra(self.loaded_dict, wavelength = self.wavelength, savefig = True)


    def new_x(self):
        Iterations = 30 #sample size for GA
        Moves_ahead = 3 #moves ahead that are calculated
        GA_iterations = 8 #times per move that the GA is used
        n_samples = 15 #sample size
        seed = 3
        self.loaded_dict = nth_iteration(self.loaded_dict, seed = seed, Iterations = Iterations, Moves_ahead = Moves_ahead, GA_iterations = GA_iterations, n_samples = n_samples)
        self.loaded_dict['next_gen_conc'][-1,:] = self.loaded_dict['best_conc_array'][:-1] #Best guess
        self.loaded_dict['next_gen_conc'][-2,:] = self.loaded_dict['best_candidate_array'][-1, 0:-1] #Best sample so far
        self.loaded_dict['best_conc_array'][:-1]
        self.conc_array = self.loaded_dict['next_gen_conc']
        self.itr =self.itr + 1
        np.save('x_values/new_x_' + str(self.itr) + '.npy', self.conc_array)

    def load_data(self, name):
        #IMPORT SPECTRA
        df = load_df(name)
        #df = subtract_baseline(df, 'F8')
        df = df.dropna(axis=1)
        df = normalize_df(df)
        #df = df.drop(['F8'], axis = 1)

        if self.itr == 0:
            df = load_df(name)
            #df = subtract_baseline(df, 'B8')
            df = normalize_df(df)
            #df = df.drop(['B8'], axis = 1)
            self.current_gen_spectra = np.asarray(df)
            self.wavelength = self.current_gen_spectra[:,0]
            self.spectra_array = self.current_gen_spectra[:,1:].T
            # INITIALIZE EXPERIMENT AND ANALYZE ZEROTH ITERATION
            self.spectra_array = self.spectra_array.T
            self.spectra_array = mms().fit(self.spectra_array).transform(self.spectra_array).T
            self.loaded_dict = zeroth_iteration(conc_array = self.conc_array, spectra_array = self.spectra_array, desired_spectra = self.x_test)
            self.spectra_array_actual = self.spectra_array
        else:
            self.current_gen_spectra = np.asarray(df)
            self.wavelength = self.current_gen_spectra[:,0]
            self.current_gen_spectra = self.current_gen_spectra[:,1:].T
            self.conc_array_actual = np.vstack((self.conc_array_actual, self.loaded_dict['next_gen_conc']))
            self.spectra_array_actual = np.vstack((self.spectra_array_actual, self.current_gen_spectra))
            self.loaded_dict['conc_array_actual'] = self.conc_array_actual
            self.loaded_dict['spectra_array_actual'] = self.spectra_array_actual
            self.current_gen_spectra = self.current_gen_spectra.T
            self.current_gen_spectra = mms().fit(self.current_gen_spectra).transform(self.current_gen_spectra).T
            self.spectra_array_actual = self.spectra_array_actual.T
            self.spectra_array_actual = mms().fit(self.spectra_array_actual).transform(self.spectra_array_actual).T
            self.loaded_dict['current_gen_spectra'] = self.current_gen_spectra
        #PLOT FITNESS
        median_fitness_list, max_fitness_list, iteration, best_candidate_array = plot_fitness(self.loaded_dict, wavelength = self.wavelength, savefig = True)
        self.loaded_dict['best_candidate_array'] = best_candidate_array
        self.loaded_dict['median_fitness_list'] = median_fitness_list
        self.loaded_dict['max_fitness_list'] = max_fitness_list
        self.loaded_dict['current_gen_spectra'] = self.current_gen_spectra
        np.save('conc_array_actual.npy', self.loaded_dict['conc_array_actual'])
        np.save('spectra_array_actual.npy', self.loaded_dict['spectra_array_actual'])
        np.save('best_candidate_array.npy', self.loaded_dict['best_candidate_array'])
        np.save('max_fitness_list.npy', self.loaded_dict['max_fitness_list'])
        np.save('median_fitness_list.npy', self.loaded_dict['median_fitness_list'])

class multi_objective:
    def __init__(self):
        self.itr = 0
        return

    def initialize_exp(self):
        # CREATE THE FIRST ITERATION USING RANDOM SAMPLING
        np.random.seed(2)
        self.conc_array = np.random.rand(15,5)
        self.conc_array_actual = self.conc_array
        #idir = '/x_values_multiobj'
        #os.makedirs(idir)

    def load_targets(self, target_1, loc_1, target_2, loc_2):
        ####### First Target ##############
        df_desired_1 = pd.read_excel(target_1)
        #df_desired = subtract_baseline(df_desired, 'A2')
        #df_desired_1 = normalize_df(df_desired_1)
        #df_desired = df_desired.drop(['A2'], axis = 1)
        #x_test = df_desired['Target'].values.reshape(-1,1)
        self.desired_spectra_1 = df_desired_1[loc_1].values.reshape(-1,1)

        ###### Second Target ##############
        df_desired_2 = pd.read_excel(target_2)
        #df_desired = subtract_baseline(df_desired, 'A2')
        #df_desired_2 = normalize_df(df_desired_2)
        #df_desired = df_desired.drop(['A2'], axis = 1)
        #x_test = df_desired['Target'].values.reshape(-1,1)
        self.desired_spectra_2 = df_desired_2[loc_2].values.reshape(-1,1)

    def new_x(self):
        Iterations = 25 #sample size for GA
        Moves_ahead = 3 #moves ahead that are calculated
        GA_iterations = 8 #times per move that the GA is used
        n_samples = 15 #sample size
        seed = 2
        self.loaded_dict = nth_iteration(self.loaded_dict, seed = seed, Iterations = Iterations, Moves_ahead = Moves_ahead, GA_iterations = GA_iterations, n_samples = n_samples)
        self.loaded_dict['next_gen_conc'][-1,:] = self.loaded_dict['best_conc_array'][:-1]
        self.loaded_dict['next_gen_conc'][-2,:] = self.loaded_dict['best_candidate_array'][-1, 0:-1]
        self.conc_array = self.loaded_dict['next_gen_conc']
        self.itr = self.itr + 1
        #np.save('x_values_multiobj/new_x_' + str(self.itr) + '.npy', self.conc_array)
        np.save('new_x_' + str(self.itr) + '.npy', self.conc_array)


    def load_data(self, name_1, name_2):
        if self.itr == 0:
            self.df_1 = load_df(name_1)
            #df = subtract_baseline(df, 'B8')
            self.df_1 = normalize_df(self.df_1)
            #df = df.drop(['B8'], axis = 1)
            self.spectra_1 = np.asarray(self.df_1)
            self.wavelength_1 = self.spectra_1[:,0]
            self.spectra_array_1 = self.spectra_1[:,1:].T

            self.df_2 = load_df(name_2)
            #df = subtract_baseline(df, 'B8')
            #df_2 = normalize_df(df_2)
            #df = df.drop(['B8'], axis = 1)
            self.spectra_2 = np.asarray(self.df_2)
            self.wavelength_2 = self.spectra_2[:,0]
            self.spectra_array_2 = self.spectra_2[:,1:].T
            self.loaded_dict = zeroth_iteration(conc_array = self.conc_array, spectra_array_1 = self.spectra_array_1, desired_spectra_1 = self.desired_spectra_1, spectra_array_2 = self.spectra_array_2, desired_spectra_2 = self.desired_spectra_2)
        else:
            self.df_1 = load_df(name_1)
            #df = subtract_baseline(df, 'B8')
            self.df_1 = normalize_df(self.df_1)
            #df = df.drop(['B8'], axis = 1)
            self.spectra_1 = np.asarray(self.df_1)
            self.wavelength_1 = self.spectra_1[:,0]
            self.spectra_array_1 = self.spectra_1[:,1:].T
            self.df_2 = load_df(name_2)
            #df = subtract_baseline(df, 'B8')
            #df_2 = normalize_df(df_2)
            #df = df.drop(['B8'], axis = 1)
            self.spectra_2 = np.asarray(self.df_2)
            self.wavelength_2 = self.spectra_2[:,0]
            self.spectra_array_2 = self.spectra_2[:,1:].T
            self.loaded_dict['conc_array_actual'] = np.vstack((self.loaded_dict['conc_array_actual'],self.loaded_dict['next_gen_conc']))
            self.loaded_dict['spectra_array_actual_1'] = np.vstack((self.loaded_dict['spectra_array_actual_1'], self.spectra_array_1))
            self.loaded_dict['spectra_array_actual_2'] = np.vstack((self.loaded_dict['spectra_array_actual_2'], self.spectra_array_2))
            self.loaded_dict['spectra_array_1'] = self.spectra_array_1
            self.loaded_dict['spectra_array_2'] = self.spectra_array_2
        median_fitness_list, max_fitness_list, iteration, best_candidate_array = plot_fitness(self.loaded_dict, savefig = False)
        self.loaded_dict['best_candidate_array'] = best_candidate_array
        self.loaded_dict['median_fitness_list'] = median_fitness_list
        self.loaded_dict['max_fitness_list'] = max_fitness_list
        #np.save('loaded_dict.npy', self.loaded_dict)
        np.save('conc_array_actual.npy', self.loaded_dict['conc_array_actual'])
        np.save('spectra_array_actual_1.npy', self.loaded_dict['spectra_array_actual_1'])
        np.save('spectra_array_actual_2.npy', self.loaded_dict['spectra_array_actual_2'])
        np.save('best_candidate_array.npy', self.loaded_dict['best_candidate_array'])
        np.save('max_fitness_list.npy', self.loaded_dict['max_fitness_list'])
        np.save('median_fitness_list.npy', self.loaded_dict['median_fitness_list'])



    def plot_results(self):
        plot_spectra(self.loaded_dict, wavelength_1 = self.wavelength_1, wavelength_2 = self.wavelength_2, savefig = False)
