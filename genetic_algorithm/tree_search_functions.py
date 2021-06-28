import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.gaussian_process import GaussianProcessRegressor
from GA_functions import fitness, GA_algorithm_unnormalized, \
                         perform_iteration, set_seed


def zeroth_iteration(**kwargs):
    '''
    Performs the zeroth iteration.
    Inputs:
    - conc_array: 2d array of the first concentration values to test
    - spectra_array: 2d array of the spectra values measured from conc_array
    - desired: 1d array of the desired spectra
    Outputs:
    - next_gen_conc: 2d array of the next concentrations to test
    - current_gen_spectra: same thing as spectra_array
    - median_fitness_list: list of the median fitness values
    - max_fitness_list: list of the max fitness values
    - iteration: list of the iteration numbers
    - mutation_rate_list: list of the mutation rates
    - mutation_rate_list_2: list of the second type of mutation rate
    '''
    
    if 'conc_array' and 'spectra_array' and 'desired_spectra' in kwargs.keys() and 'single_data' not in kwargs.keys():
        conc_array = kwargs['conc_array']
        spectra_array = kwargs['spectra_array']
        desired_spectra = kwargs['desired_spectra']
        setting = 1
    elif 'single_data' and 'single_desired' and 'conc_array' in kwargs.keys() and 'spectra_array' not in kwargs.keys():
        single_data = kwargs['single_data']
        single_desired = kwargs['single_desired']
        conc_array = kwargs['conc_array']
        setting = 2 
    elif 'single_data' and 'spectra_array' in kwargs.keys():
        conc_array = kwargs['conc_array']
        spectra_array = kwargs['spectra_array']
        desired_spectra = kwargs['desired_spectra']
        single_data = kwargs['single_data']
        single_desired = kwargs['single_desired']
        setting = 3
  
    seed = np.random.randint(0, 100)
    set_seed(seed)
    # spectra = spectra.T
    # spectra = MinMaxScaler().fit(spectra).transform(spectra).T
    # desired = MinMaxScaler().fit(x_test).transform(x_test).T
    # desired = desired.reshape(1, -1)[0].reshape(-1, 1).T
    median_fitness_list = []
    max_fitness_list = []
    iteration = []
    mutation_rate_list = []
    fitness_multiplier_list = []
    # Calculate Fitness

    if setting == 1:
        a, median_fitness, max_fitness = fitness(spectra = spectra_array, 
                                                 conc = conc_array, 
                                                 desired_spectra = desired_spectra)
        current_gen_spectra = spectra_array
    elif setting == 2:
        a, median_fitness, max_fitness = fitness(single_data = single_data, 
                                                 conc = conc_array, 
                                                 single_desired = single_desired)
    elif setting == 3:
        a, median_fitness, max_fitness = fitness(spectra = spectra_array, 
                                                 conc = conc_array, 
                                                 desired_spectra = desired_spectra,
                                                 single_data = single_data,
                                                 single_desired = single_desired)
        
    # Appending to list
    i = 0
    median_fitness_list.append(median_fitness)
    max_fitness_list.append(max_fitness)
    iteration.append(i)
    next_gen_conc = conc_array
    # Plotting
    plt.scatter(iteration, median_fitness_list, label='median fitness')
    plt.scatter(iteration, max_fitness_list, label='max fitness')
    plt.ylim([0, 3])
    plt.legend(loc=2)
    plt.show()
    print('The max fitness is:', max_fitness)
    print('The median fitness is:', median_fitness)
    
    if setting == 1:
            output_dict = {'next_gen_conc':next_gen_conc,
                   'current_gen_spectra':current_gen_spectra,
                   'median_fitness_list':median_fitness_list,
                   'max_fitness_list':max_fitness_list,
                   'iteration':iteration,
                   'mutation_rate_list':mutation_rate_list,
                   'mutation_rate_2_list':fitness_multiplier_list,
                   'conc_array':conc_array,
                   'spectra_array':spectra_array,
                   'desired_spectra':desired_spectra,
                   'conc_array_actual': next_gen_conc,
                   'spectra_array_actual': current_gen_spectra}
    elif setting == 2:
            output_dict = {'next_gen_conc':next_gen_conc,
                   'single_data':single_data,
                   'median_fitness_list':median_fitness_list,
                   'max_fitness_list':max_fitness_list,
                   'iteration':iteration,
                   'mutation_rate_list':mutation_rate_list,
                   'mutation_rate_2_list':fitness_multiplier_list,
                   'conc_array':conc_array,
                   'single_desired':single_desired,
                   'conc_array_actual': next_gen_conc,
                   'single_data_actual': single_data}
    elif setting == 3:
            output_dict = {'next_gen_conc':next_gen_conc,
                   'current_gen_spectra':current_gen_spectra,
                   'single_data':single_data,
                   'single_desired':single_desired,
                   'median_fitness_list':median_fitness_list,
                   'max_fitness_list':max_fitness_list,
                   'iteration':iteration,
                   'mutation_rate_list':mutation_rate_list,
                   'mutation_rate_2_list':fitness_multiplier_list,
                   'conc_array':conc_array,
                   'spectra_array':spectra_array,
                   'desired_spectra':desired_spectra,
                   'conc_array_actual': next_gen_conc,
                   'spectra_array_actual': current_gen_spectra,
                   'single_data_actual': single_data}
    return output_dict

    
def perform_Surrogate_Prediction(next_gen_conc,
                                 conc_array_actual,
                                 spectra_array_actual):
    '''
    Fits a surrogate model to conc_array_actual and
    spectra_array_actual and predicts the spectra of next_gen_conc
    Inputs:
    - next_gen_conc: A 2d array of the concentrations from
    the current iteration
    - conc_array_actual: A 2d array of the concentrations from
    all the previous iterations
    - spectra_array_actual: A 2d array of all the concentrations
    from all the previous iterations.
    Outputs:
    - spectra_prediction: A 2d array of the predicted spectra of
    next_gen_conc
    - score: The score of the surrogate model
    '''
    gpr = GaussianProcessRegressor().fit(
        conc_array_actual, spectra_array_actual)
    score = gpr.score(conc_array_actual,
                      spectra_array_actual)
    spectra_prediction = gpr.predict(next_gen_conc)
    return spectra_prediction, score


def MCTS(MCTS_dict):
    if 'current_gen_spectra' and 'spectra_array_actual' in MCTS_dict.keys() and 'single_data' not in MCTS_dict.keys():
        Iterations_per_move = MCTS_dict['Iterations']
        moves = MCTS_dict['moves']
        GA_iterations = MCTS_dict['GA_iterations']
        n_samples = MCTS_dict['n_samples']
        current_gen_spectra = MCTS_dict['current_gen_spectra']
        next_gen_conc = MCTS_dict['next_gen_conc']
        desired_spectra = MCTS_dict['desired_spectra']
        conc_array_actual = MCTS_dict['conc_array_actual']
        spectra_array_actual = MCTS_dict['spectra_array_actual']
        seed = MCTS_dict['seed']
        setting = 1
    elif 'single_data' and 'single_desired' in MCTS_dict.keys() and 'current_gen_spectra' not in MCTS_dict.keys():
        Iterations_per_move = MCTS_dict['Iterations']
        moves = MCTS_dict['moves']
        GA_iterations = MCTS_dict['GA_iterations']
        n_samples = MCTS_dict['n_samples']
        single_data = MCTS_dict['single_data']
        next_gen_conc = MCTS_dict['next_gen_conc']
        single_desired = MCTS_dict['single_desired']
        conc_array_actual = MCTS_dict['conc_array_actual']
        single_data_actual = MCTS_dict['single_data_actual']
        seed = MCTS_dict['seed']
        setting = 2
    elif 'single_data' and 'single_desired' and 'current_gen_spectra' in MCTS_dict.keys():
        Iterations_per_move = MCTS_dict['Iterations']
        moves = MCTS_dict['moves']
        GA_iterations = MCTS_dict['GA_iterations']
        n_samples = MCTS_dict['n_samples']
        single_data = MCTS_dict['single_data']
        desired_spectra = MCTS_dict['desired_spectra']
        conc_array_actual = MCTS_dict['conc_array_actual']
        single_data_actual = MCTS_dict['single_data_actual']
        current_gen_spectra = MCTS_dict['current_gen_spectra']
        next_gen_conc = MCTS_dict['next_gen_conc']
        single_desired = MCTS_dict['single_desired']
        single_data_actual = MCTS_dict['single_data_actual']
        seed = MCTS_dict['seed']
        setting = 3

    '''Performs another an additional iteration. Use this after
    performing the 0th iteration with the function zeroth_iteration
      Inputs:
      - Iterations_per_move: An integer of how many iterations
      per move should be performed.
      - moves: An integer of how many moves ahead to predict.
      - GA_iterations: An integer of how many GA iterations
      to perform.
      - current_gen_spectra: 2d array of the all the spectra where
      the number of rows equal the number of samples and the number
      of columns equal the number of datapoints in the spectra. It
      should have the same number of rows as next_gen_conc.
      - next_gen_conc: 2d array of the concentrations used to create
      current_gen_spectra.
      It should have the same number of rows as the n_samples.
      - x_test: 1d array of the desired spectra. It should have the
      same length as the spectra data from current_gen_spectra.
      - conc_array_actual: A 2d array of all the conc_arrays from
      the previous generations.
      - spectra_array_actual: A 2d array of all the spectra arrays
      from the previous
      generations/iterations.
      - seed: An integer which determines the random seed used.
      - n_samples: An integer of how many samples per
      batch/iteration/generations.
      Outputs:
      - mutation_rate: mutation rate used in the next iteration
      - mutation_rate_2: mutation rate 2 used in the next iteration
      - best_move: A 1d array of the best move that should be taken
      and the fitness values corresponding to these moves.
      - best_move_turn: An integer of when the best move will occur.
      - max_fitness: A float of the fitness value corresponding to
      the best move.
      - surrogate_score: A float of the score of the surrogate model.
      - desired: normalized x_test
      - current_gen_spectra: 2d array of the current generation of
      spectra values.
    '''
    next_gen_conc_original = next_gen_conc
    dictionary_of_moves = {}
    fitness_array = []  # initialize array for flake8
    # move_fitness_list = []
    # moves_list = []
    for move_number in range(moves):
        conc_fitness_list = []
        for GA_iteration in range(GA_iterations):
            if GA_iteration == 0:
                for cols in range(move_number+1):
                    mutation_rate_array = (np.round(
                        np.random.uniform(
                            0, 10, Iterations_per_move))/10
                                          ).reshape(-1, 1)
                    fitness_multiplier_array = (np.round(
                        np.random.uniform(
                            0, 10, Iterations_per_move))/10
                                               ).reshape(-1, 1)
                    move = np.hstack((mutation_rate_array,
                                      fitness_multiplier_array))
                    if cols == 0:
                        move_array = move
                    else:
                        move_array = np.hstack((move_array, move))
            else:
                optimize_array = np.array(
                    [10]*Iterations_per_move).reshape(-1, 1)
                move_array, _, _ = GA_algorithm_unnormalized(
                        fitness_array[:, move_number].reshape(-1, 1),
                        move_array, optimize_array, 50,
                        Iterations_per_move, 0.1, 2)
            if move_number == 0:
                all_moves_array = move_array
            else:
                all_moves_array = np.hstack((all_moves_array, move_array))
            for row in range(Iterations_per_move):
                Fitness_move_1 = []
                for J in range(0, 2*(move_number+1), 2):
                    if J == 0:
                        next_gen_conc = next_gen_conc_original
                    if setting == 1:     
                        loaded_dict = {'mutation_rate':move_array[row, J],
                                       'mutation_rate_2':move_array[row, J+1],
                                       'n_parents':50,
                                       'n_samples':n_samples,
                                       'next_gen_conc':next_gen_conc,
                                       'desired_spectra':desired_spectra,
                                       'current_gen_spectra':current_gen_spectra}
                        next_gen_conc, median_fitness, max_fitness = \
                            perform_iteration(loaded_dict)
                    elif setting == 2:
                        loaded_dict = {'mutation_rate':move_array[row, J],
                                       'mutation_rate_2':move_array[row, J+1],
                                       'n_parents':50,
                                       'n_samples':n_samples,
                                       'next_gen_conc':next_gen_conc,
                                       'single_desired':single_desired,
                                       'single_data':single_data}
                        next_gen_conc, median_fitness, max_fitness = \
                            perform_iteration(loaded_dict)
                    elif setting == 3:
                        loaded_dict = {'mutation_rate':move_array[row, J],
                                       'mutation_rate_2':move_array[row, J+1],
                                       'n_parents':50,
                                       'n_samples':n_samples,
                                       'next_gen_conc':next_gen_conc,
                                       'single_desired':single_desired,
                                       'single_data':single_data,
                                       'desired_spectra':desired_spectra,
                                       'current_gen_spectra':current_gen_spectra}
                        next_gen_conc, median_fitness, max_fitness = \
                            perform_iteration(loaded_dict)
                        
                    if setting == 1:
                        ############################## NORMALIZATION #################################################
                        spectra_array_actual = spectra_array_actual.T
                        spectra_array_actual = MinMaxScaler().fit(spectra_array_actual).transform(spectra_array_actual).T
                        ##############################################################################################
                        simulated_spectra, surrogate_score = \
                            perform_Surrogate_Prediction(
                                                     next_gen_conc,
                                                     conc_array_actual,
                                                     spectra_array_actual)
                        ############################## NORMALIZATION ########################################################
                        simulated_spectra = simulated_spectra.T
                        simulated_spectra = MinMaxScaler().fit(simulated_spectra).transform(simulated_spectra).T
                        ######################################################################################################
                        desired_spectra = desired_spectra.reshape(1,-1)
                        conc_fitness, median_fitness, max_fitness = fitness(
                            spectra = simulated_spectra, conc = next_gen_conc, desired_spectra = desired_spectra)
                        best_conc = conc_fitness[-1, simulated_spectra.shape[1]:]

                    elif setting == 2:
                        simulated_single_data, surrogate_score = \
                            perform_Surrogate_Prediction(
                                                         next_gen_conc,
                                                         conc_array_actual,
                                                         single_data_actual)
                        conc_fitness, median_fitness, max_fitness = fitness(single_data = single_data, 
                                                                     conc = next_gen_conc,
                                                                     single_desired = single_desired)
                        best_conc = conc_fitness[-1, simulated_single_data.shape[1]:]
  
                    elif setting == 3:
                        simulated_single_data, surrogate_score = \
                            perform_Surrogate_Prediction(
                                                     next_gen_conc,
                                                     conc_array_actual,
                                                     single_data_actual)
                    conc_fitness_list.append(best_conc)
                    Fitness_move_1.append(max_fitness)
                    conc_fitness_array = np.asarray(conc_fitness_list)
                    if move_number == 0:
                        best_conc_array = conc_fitness_array
                    else:
                        best_conc_array = np.vstack((best_conc_array,
                                                     conc_fitness_array))
                fitness_list = np.asarray(Fitness_move_1).reshape(1, -1)
                if row == 0:
                    fitness_array = fitness_list
                else:
                    fitness_array = np.vstack((fitness_array, fitness_list))
            move_fitness = np.hstack((move_array, fitness_array))
            if GA_iteration == 0:
                move_fitness_all = move_fitness
            else:
                move_fitness_all = np.vstack((move_fitness_all, move_fitness))
            dictionary_of_moves.update({move_number + 1: move_fitness_all})
    best_move_list = []
    for m in range(moves):
        dictionary_of_moves[moves]
        row_number_max_fitness = np.unravel_index(
            np.argmax(dictionary_of_moves[m+1][:, -1]),
            dictionary_of_moves[m+1][:, -1].shape)
        best_move = dictionary_of_moves[m+1][row_number_max_fitness, :]
        best_move_list.append(best_move[-1][-1])
    best_move_array = np.asarray(best_move_list)
    best_move_number = np.unravel_index(np.argmax(best_move_array),
                                        best_move_array.shape)
    best_move_number = best_move_number[0] + 1
    dictionary_of_moves[best_move_number]
    row_number_max_fitness = np.unravel_index(
        np.argmax(dictionary_of_moves[best_move_number][:, -1]),
        dictionary_of_moves[best_move_number][:, -1].shape)
    best_play = \
        dictionary_of_moves[best_move_number][row_number_max_fitness, :]
    max_fitness = best_play[0][-1]
    mutation_rate = best_play[0][0]
    fitness_multiplier = best_play[0][1]
    
    
    return mutation_rate, fitness_multiplier, best_play, \
        best_move_number, max_fitness, surrogate_score, \
        best_conc_array, \
        dictionary_of_moves


def nth_iteration(loaded_dict, **kwargs):
    '''
       Performs another an additional iteration. Use this after performing
       the 0th iteration with the function zeroth_iteration.
      Inputs:
      - Iterations: An integer of how many iterations the MCTS will perform
      per batch.
      - Moves_ahead: An integer of how many moves ahead to predict.
      - GA_iterations: An integer of how many GA iterations to perform.
      - n_samples: An integer of how many samples per
        batch/iteration/generations.
      - current_gen_spectra: 2d array of the all the spectra where the number
      of rows equal the number of samples and the number of columns equal
      the number of datapoints in the spectra.It should have the same number
      of rows as next_gen_conc.
      - next_gen_conc: 2d array of the concentrations used to create
      current_gen_spectra. It should have the same number of rows as
      the n_samples.
      - x_test: 1d array of the desired spectra. It should have the same length
      as the spectra data from current_gen_spectra
      - conc_array_actual: A 2d array of all the conc_arrays from the
      previous generations.
      - spectra_array_actual: A 2d array of all the spectra arrays from
      the previous generations/iterations.
      - seed: An integer which determines the random seed used.
      - median_fitness_list: A list of all median fitness values of previous
      generations/iterations.
      - max_fitness_list: A list of all max fitness values of previous
      generations/iterations.
      - iteration: A list of all the iterations from the previous
      generations/iterations.
      - mutation_rate_list: a list of all the mutation rates used in
      previous generation/iterations.
      - mutation_rate_list_2: a list of all the mutation2 rates used
      in the previous generations/iterations.
      Outputs:
      - mutation_rate: mutation rate used in the next iteration
      - mutation_rate_2: mutation rate 2 used in the next iteration
      - mutation_rate_list: an updated list of all the mutation
      rates used in previous generation/iterations.
      - mutation_rate_list_2: an updated list of all the mutation2 rates
      used in the previous generations/iterations.
      - best_move: A 1d array of the best move that should be taken and
      the fitness values corresponding to these moves.
      - best_move_turn: An integer of when the best move will occur.
      - max_fitness: A float of the fitness value corresponding to the
      best move.
      - surrogate_score: A float of the score of the surrogate model.
      - next_gen_conc: A 2d array of the next generation of
      concentrations to be tested using uv-vis.
    '''
    seed = kwargs['seed']
    Iterations = kwargs['Iterations']
    Moves_ahead = kwargs['Moves_ahead']
    GA_iterations = kwargs['GA_iterations']
    n_samples = kwargs['n_samples']        
    set_seed(seed)
    
    loaded_dict.update({'Iterations':Iterations})
    loaded_dict.update({'moves':Moves_ahead})
    loaded_dict.update({'GA_iterations':GA_iterations})
    loaded_dict.update({'seed':seed})
    loaded_dict.update({'n_samples':n_samples})
    
    mutation_rate, mutation_rate_2, best_move, best_move_turn, \
        max_fitness, surrogate_score, \
        best_conc_array, \
        dictionary_of_moves = MCTS(loaded_dict)
    print('The best move has a fitness value of', max_fitness)
    print('The best move occurs in', best_move_turn, 'turns.')
    print()
    print('The surrogate model has a score of:', surrogate_score)
    print()
    mutation_rate_list = loaded_dict['mutation_rate_list']
    mutation_rate_2_list = loaded_dict['mutation_rate_2_list']                                 
    mutation_rate_list.append(mutation_rate)
    mutation_rate_2_list.append(mutation_rate_2)
    loaded_dict.update({'mutation_rate':mutation_rate})
    loaded_dict.update({'mutation_rate_2':mutation_rate_2})
    loaded_dict.update({'mutation_rate_list':mutation_rate_list})
    loaded_dict.update({'mutation_rate_2_list':mutation_rate_2_list})
    
    ###################### NORMALIZATION #############################
    #current_gen_spectra = loaded_dict['current_gen_spectra']
    #current_gen_spectra = current_gen_spectra.T
    #current_gen_spectra = MinMaxScaler().fit(current_gen_spectra). \
    #    transform(current_gen_spectra).T
    #loaded_dict.update({'current_gen_spectra':current_gen_spectra})
    ###################################################################
    
    if 'current_gen_spectra' and 'desired_spectra' in loaded_dict.keys() and 'single_data' not in loaded_dict.keys():
        setting = 1
        next_gen_conc, median_fitness, max_fitness = perform_iteration(loaded_dict)
    elif 'single_data' and 'single_desired' in loaded_dict.keys() and 'current_gen_spectra' not in loaded_dict.keys():
        setting = 2
        next_gen_conc, median_fitness, max_fitness = perform_iteration(loaded_dict)
    elif 'single_data' and 'single_desired' and 'current_gen_spectra' in loaded_dict.keys():
        setting = 3
        next_gen_conc, median_fitness, max_fitness = perform_iteration(loaded_dict)

    best_conc_array = \
        best_conc_array[np.argsort(best_conc_array[:, -1])][-1, :]
    print(next_gen_conc)
    loaded_dict.update({'next_gen_conc':next_gen_conc})
    loaded_dict.update({'median_fitness':median_fitness})
    loaded_dict.update({'max_fitness':max_fitness})
    loaded_dict.update({'best_conc_array':best_conc_array})
    loaded_dict.update({'best_move':best_move})
    loaded_dict.update({'best_move_turn':best_move_turn})
    loaded_dict.update({'dictionary_of_moves':dictionary_of_moves})
    
#    if setting == 1:
#        loaded_dict.update({current_gen_spectra})
        
#         output_dict = {'mutation_rate':mutation_rate,
#                        'mutation_rate_2':mutation_rate_2,
#                        'mutation_rate_list':mutation_rate_list,
#                        'mutation_rate_2_list':mutation_rate_2_list,
#                        'best_move':best_move,
#                        'best_move_turn':best_move_turn,
#                        'max_fitness':max_fitness,
#                        'surrogate_score':surrogate_score,
#                        'next_gen_conc':next_gen_conc,
#                        'best_conc_array':best_conc_array,
#                        'dictionary_of_moves':dictionary_of_moves,
#                        'conc_array':next_gen_conc,
#                        'spectra_array':current_gen_spectra,
#                        'desired':x_test,
#                        'conc_array_actual': conc_array_actual,
#                        'spectra_array_actual': spectra_array_actual,
#                        'current_gen_spectra': current_gen_spectra,
#                        'iteration':iteration,
#                        'median_fitness_list':median_fitness_list,
#                        'max_fitness_list':max_fitness_list}
    return loaded_dict


#def plot_fitness(next_gen_conc, current_gen_spectra, x_test,
#                 median_fitness_list, max_fitness_list, iteration, savefig):
def plot_fitness(loaded_dict, **kwargs):
    median_fitness_list = loaded_dict['median_fitness_list']
    max_fitness_list = loaded_dict['max_fitness_list']
    iteration = loaded_dict['iteration']
    savefig = kwargs['savefig']
    next_gen_conc = loaded_dict['next_gen_conc']
    
    if 'current_gen_spectra' in loaded_dict.keys() and 'single_data' not in loaded_dict.keys():
        spectra = loaded_dict['current_gen_spectra']
        desired_spectra = loaded_dict['desired_spectra']
        a, median_fitness, max_fitness = fitness(spectra = spectra,
                                             conc = next_gen_conc,
                                             desired_spectra = desired_spectra)
        setting = 1
    elif 'single_data' in loaded_dict.keys() and 'current_gen_spectra' not in loaded_dict.keys():
        single_data = loaded_dict['single_data']
        single_desired = loaded_dict['single_desired']
        array, median_fitness, max_fitness = fitness(single_data = single_data, 
                                                 conc = next_gen_conc,
                                                 single_desired = single_desired)
        setting = 2
    elif 'single_data' and 'current_gen_spectra' in loaded_dict.keys():
        single_data = loaded_dict['single_data']
        current_gen_spectra = loaded_dict['current_gen_spectra']
        single_desired = loaded_dict['single_desired']
        desired_spectra = loaded_dict['desired_spectra']
        setting = 3
    
    '''
        Plots the fitness of the generations with the iteration number
        Inputs:
        - next_gen_conc: 2d array of the concentrations used to make
        the current_gen_spectra.
        - current_gen_spectra: 2d array of the all the spectra where the number
        of rows equal the number of samples and the number of columns
        equal the number of datapoints in the spectra. It should have the
        same number of rows
          as next_gen_conc.
        - x_test: 1d array of the desired spectra. It should have the
        same length as the spectra data from cuurent_gen_spectra
        - median_fitness_list: list of median fitness of all the
        previous iterations.
        - max_fitness_list: list of max fitness of all the previous iterations.
        - iteration: list of the iteration numbers
        - savefig: either True or False, determines whether to save the
        figures as images.
        Outputs:
        - median_fitness_list: updates median fitness list
        - max_fitness_list: updates max fitness lits
        - iteration: updates iteration number
    '''

    # Appending to list
    i = iteration[-1]
    i = i + 1
    median_fitness_list.append(median_fitness)
    max_fitness_list.append(max_fitness)
    iteration.append(i)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iteration, median_fitness_list, label='median fitness')
    ax.plot(iteration, max_fitness_list, label='max fitness')
    ax.set_xticks(iteration)
    ax.set_ylabel('fitness')
    ax.set_xlabel('Iteration')
    ax.legend()
    print('The max fitness is:', max_fitness)
    print('The median fitness is:', median_fitness)
    if savefig is True:
        plt.savefig('Plot_Fitness.png')
    else:
        pass
    return median_fitness_list, max_fitness_list, iteration


def plot_spectra(loaded_dict, **kwargs):
    '''
        Plots the Uv-vis spectra along with the desired one.
        Inputs:
        - current_gen_spectra: 2d array of the all the spectra where the
        number of rows equal the number of samples and the number of columns
        equal the number of datapoints in the spectra.
        - x_test: 1d array of the desired spectra. It should have the same
        length as the spectra data from cuurent_gen_spectra.
        - wavelength: 1d array of the values of the wavelength corresponding
        to the values for the spectra data.
        - iteration: list of the iteration numbers
        - savefig: either True or False, determines whether to save the figures
        as images.
    '''
    wavelength = kwargs['wavelength']
    savefig = kwargs['savefig']
    spectra = loaded_dict['current_gen_spectra']
    desired_spectra = loaded_dict['desired_spectra'].reshape(1,-1)
    iteration = loaded_dict['iteration']
    # spectra = spectra.T
    # spectra = MinMaxScaler().fit(spectra).transform(spectra).T
    # desired = MinMaxScaler().fit(x_test).transform(x_test).T
    # desired = desired.reshape(1, -1)[0].reshape(-1, 1).T
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))
    ax[0].plot(wavelength, desired_spectra.T, label='Target', linewidth=5, c='k')
    ax[0].set_title('Spectra of All Samples')
    ax[0].set_ylabel('Absorbance')
    ax[0].set_xlabel('Wavelength (nm)')
    for ii in range(spectra.shape[0]):
        ax[0].plot(wavelength, spectra[ii, :])
    ax[0].legend(loc=2)
    fitness_list = []
    for ii in range(spectra.shape[0]):
        fitness = 1/np.sum(np.abs(spectra[ii, :] - desired_spectra))
        fitness_list.append(fitness)
    fitness_array = np.asarray(fitness_list).reshape(-1, 1)
    array = np.hstack((spectra, fitness_array))
    sorted_array = array[np.argsort(array[:, -1])]
    ax[1].plot(wavelength, desired_spectra.T, label='Target',
               linewidth=5, c='k')
    ax[1].plot(wavelength, sorted_array[-1, :-1],
               label='Best Sample', linewidth=3)
    ax[1].set_title('Spectra of Best Sample')
    ax[1].set_ylabel('Absorbance')
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].legend(loc=2)
    figure_name = 'Iteration_' + str(iteration[-1]) + '.png'
    if savefig is True:
        plt.savefig(figure_name)
    else:
        pass

def plot_DLS(loaded_dict, **kwargs):
    '''
        Plots the Uv-vis spectra along with the desired one.
        Inputs:
        - current_gen_spectra: 2d array of the all the spectra where the
        number of rows equal the number of samples and the number of columns
        equal the number of datapoints in the spectra.
        - x_test: 1d array of the desired spectra. It should have the same
        length as the spectra data from cuurent_gen_spectra.
        - wavelength: 1d array of the values of the wavelength corresponding
        to the values for the spectra data.
        - iteration: list of the iteration numbers
        - savefig: either True or False, determines whether to save the figures
        as images.
    '''
    savefig = kwargs['savefig']
    single_data = loaded_dict['single_data']
    single_desired = loaded_dict['single_desired']
    iteration = loaded_dict['iteration']
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.hist(single_data, rwidth=0.9, label='Samples')
    plt.hist([single_desired]*int(len(single_data)/4), rwidth=0.1, label='Target')
    plt.legend()
    plt.show()
    figure_name = 'Iteration_' + str(iteration[-1]) + '.png'
    if savefig is True:
        plt.savefig(figure_name)
    else:
        pass
