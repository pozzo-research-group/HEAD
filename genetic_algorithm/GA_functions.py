import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms


def set_seed(x):
    global seed
    seed = x


def conc_to_spectra(conc, stock_spectra):
    ''' Simulates Beer's law and with concentrations of differnt stocks
    and the spectra of these stocks.
        Inputs:
       - conc: A 2-D array with the number of columns equal to the number
       of stocks and rows equal to the number of different combinations of
       stocks to be tested.
       - stock_spectra: A 2-D array with the number of columns equal to the
       number of stocks and the rows equal to the number of absorabance
       datapoints.
        Outputs:
       - spectra_final: A 2-D array with the number of columns equal to the
       number of absorbance datapoints and the number of rows equal to the
       number of combinations of stocks that were tested.
    '''
    for i in range(conc.shape[0]):
        for j in range(stock_spectra.shape[1]):
            spectra_part = conc[i, j]*stock_spectra[:, j]
            if j == 0:
                spectra = spectra_part
            else:
                spectra = np.vstack((spectra, spectra_part))
        spectra_row = np.sum(spectra, axis=0)
        if i == 0:
            spectra_final = spectra_row
        else:
            spectra_final = np.vstack((spectra_final, spectra_row))
    return spectra_final


def prepare_desired_spectra(x_test):
    x_test = mms().fit(x_test).transform(x_test).T
    x_test = x_test.reshape(1, -1)[0].reshape(-1, 1).T
    return x_test


def fitness(spectra, conc, desired):
    '''Sorts an array by its fitness with the most fit row at the bottom
    of the array.
       Inputs:
     - spectra: A 2D array of spectra values dimensionally reduced to 3
     columns by pca.
     - conc: A 2D array of the normalized concentrations used to create
     the spectra.
     It should have 3 columns and the same number of rows as the spectra
     array.
     - desired: A 2D array of 1 row and 3 columns, with each column
     representing the desired dimensionally reduced spectra value.
     The fitness value is determined by how close the other spectras
     are to this one.
       Outputs:
     - sorted_array: A 2D array with 7 columns and the same number
     of rows as the inputs "spectra" and "conc". Columns 0-2 are
     the spectra, columns 3-6 are the concentrations, and column 7
     is the fitness score. The rows are sorted so that the most fit
     row is at the bottm of the array.
       '''
    np.random.seed(seed)
    fit_score_array = []
    for i in range(spectra.shape[0]):
        fit_score = 1/np.sum(np.abs(spectra[i, :] - desired)+0.0001)
        fit_score_array.append(fit_score)
    fitness_score = np.asarray(fit_score_array)
    new_array = np.hstack((spectra, conc, fitness_score.reshape(1, -1).T))
    sorted_array = new_array[np.argsort(new_array[:, -1])]
    lower_fitness, upper_fitness = np.array_split(sorted_array, 2)
    return upper_fitness, np.median(fitness_score), np.max(fitness_score)


def select_parents(sorted_array, n_parents):
    '''Randomly selects parents, the ones with a higher fitness will have
    a higher chance of being selected. Uses a roulette wheel approach
    where the probability of being selected is proportional to the fitness.
       Inputs:
     - sorted_array: 2D array of 7 columns. Columns 0-2 are the spectra
     after being dimensionally reduced to 3 columns (pca). Columns 3-6
     are the normalized concentrations. Column 7 is the fitness of the row.
     - n_parents determines how many parents are created.
       Outputs:
     - parents: 2D array of the same number of columns as sorted_array, but
     with n_parents rows. Rows with higher fitness should appear in a higher
     frequency in this row than ones with lower fitness.'''
    np.random.seed(seed)
    fitness_list = sorted_array[:, -1]
    fitness_sum = np.sum(fitness_list)
    probability = fitness_list/fitness_sum
    cumsum = np.cumsum(probability)
    for itr in range(n_parents):
        rand_num = np.random.rand()
        for i in range(cumsum.shape[0]):
            if cumsum[i] > rand_num:
                UB = cumsum[i]
                if i == 0:
                    LB = cumsum[i]
                    break
                else:
                    LB = cumsum[i-1]
                    break
        if itr == 0:
            parents = sorted_array[i]
            LB = LB
            LB = UB
        else:
            parents = np.vstack((parents, sorted_array[i]))
    return parents


def crossover(parents, n_offspring):
    '''
       Performs a crossover between the parents to create offspring that
       have characteristcs of both parents.
       Inputs:
     - parents: A 2D array of 3 columns, representing the concentrations,
     and n_parents rows.
     - n_offspring: An integer representing the number of offspring to be
     created from these parents.
       Outputs
     - offspring_array: A 2D array of 3 columns and n_offspring rows.
    '''
    np.random.seed(seed)
    for i in range(n_offspring):
        random_row1 = np.int(np.round(np.random.rand()*parents.shape[0]-1))
        random_row2 = np.int(np.round(np.random.rand()*parents.shape[0]-1))
        p1 = parents[random_row1, :]  # selects first parent
        p2 = parents[random_row2, :]  # selects second parent
        row_of_concs = []
        for n_stocks in range(parents.shape[1]-1):
            p1_conc = str(p1[n_stocks])
            p2_conc = str(p2[n_stocks])

            def normalize_sig_figs(p1_conc):
                if len(p1_conc) < 5:
                    p1_conc = p1_conc + '0' + '0'
                return p1_conc

            def cross_parents(p1_conc, p2_conc):
                zero = p1_conc[0]
                decimal = p1_conc[1]
                p1_digit1 = p1_conc[2]
                p1_digit2 = p1_conc[3]
                p1_digit3 = p1_conc[4]
                p2_digit1 = p2_conc[2]
                p2_digit2 = p2_conc[3]
                p2_digit3 = p2_conc[4]
                random_number = np.random.rand()
                if random_number < 0.5:
                    digit1 = p1_digit1
                else:
                    digit1 = p2_digit1
                random_number = np.random.rand()
                if random_number < 0.5:
                    digit2 = p1_digit2
                else:
                    digit2 = p2_digit2
                random_number = np.random.rand()
                if random_number < 0.5:
                    digit3 = p1_digit3
                else:
                    digit3 = p2_digit3
                offspring_conc = float(zero + decimal +
                                       digit1 + digit2 + digit3)
                return offspring_conc
            p1_conc = normalize_sig_figs(p1_conc)
            p2_conc = normalize_sig_figs(p2_conc)
            offspring_conc = cross_parents(p1_conc, p2_conc)
            row_of_concs.append(offspring_conc)
        row_of_offspring = np.asarray(row_of_concs)
        if i == 0:
            offspring = row_of_offspring
        else:
            offspring = np.vstack((offspring, row_of_offspring))
    return offspring


def mutation(array, rate):
    '''
        Performs a mutation on some of the values in the offspring array.
        It converts the value to a string and then changes one of the
        digits to a random number.
        Inputs:
      - array: A 2D array of the concentrations of the offspring. It should
      have any number or rows, and the number of dimensions as columns.
      - rate: The mutation rate. It is a float from 0 to 1, with 1 being
      a high mutation rate.
        Outputs:
      - array: returns an array with mutated values. It should have the
      same dimensions of the input array.
    '''
    np.random.seed(seed)

    def normalize_sig_figs(p1_red_conc):
        if len(p1_red_conc) < 5:
            p1_red_conc = p1_red_conc + '0' + '0'
        return p1_red_conc

    for j in range(array.shape[0]):
        for i in range(array.shape[1]):
            if np.random.rand() < rate:
                conc = str(array[j, i])
                conc = normalize_sig_figs(conc)
                column = int(np.round(np.random.uniform(2, 4)))
                random_int = str(int(np.round(np.random.uniform(0, 9))))
                if column == 2:
                    digit1 = random_int
                    digit2 = conc[3]
                    digit3 = conc[4]
                elif column == 3:
                    digit1 = conc[2]
                    digit2 = random_int
                    digit3 = conc[4]
                else:
                    digit1 = conc[2]
                    digit2 = conc[3]
                    digit3 = random_int
                mutated_conc = conc[0] + conc[1] + digit1 + digit2 + digit3
                mutated_conc = float(mutated_conc)
                array[j, i] = mutated_conc
    return array


def mutation2(array, rate):
    '''
        Performs a mutation on some of the values in the offspring array.
        It converts the value to a string and then changes one of the digits
        to a random number.
        Inputs:
      - array: A 2D array of the concentrations of the offspring. It should
      have any number or rows, and the number of dimensions as the columns.
      - rate: The mutation rate. It is a float from 0 to 1, with 1 being a
      high mutation rate.
        Outputs:
      - array: returns an array with mutated values. It should have the same
      dimensions of the input array.
    '''
    np.random.seed(seed)
    for j in range(array.shape[0]):
        for i in range(array.shape[1]):
            if np.random.rand() < rate:
                array[j, i] = array[j, i] + (np.random.rand()-0.5)/50
    return array


def GA_algorithm(x_train_spectra, y_train_conc, x_test, n_parents, n_offspring,
                 mutation_rate, mutation_rate_2):
    np.random.seed(seed)
    # Obtains fitness of the input concentrations and their spectra
    array, median_fitness, max_fitness = fitness(x_train_spectra, y_train_conc,
                                                 x_test)
    # Obtains the parents of the ordered fitness array
    parents = select_parents(array, n_parents)
    # Prepares the parents array to remove the spectra information
    parents = parents[:, parents.shape[1] - y_train_conc.shape[1]-1:]
    # Performs a crossover to obtain the offspring of the parents
    offspring = crossover(parents, n_offspring)
    # conc_offspring = offspring[:,x_train_spectra.shape[1]:-1]
    conc_offspring = offspring
    conc_offspring_unique = conc_offspring
    # conc_offspring_unique = np.unique(conc_offspring, axis=0)
    # Perfroms a mutation on the offspring array
    conc_offspring_mutated = mutation(conc_offspring_unique, mutation_rate)
    # Filters offspring array for negative values and normalizes all values
    conc_offspring_mutated = np.abs(conc_offspring_mutated)
    for j in range(conc_offspring_mutated.shape[0]):
        row_sum = np.sum(conc_offspring_mutated[j, :])
        for i in range(conc_offspring_mutated.shape[1]):
            conc_ij = conc_offspring_mutated[j, i]
            conc_offspring_mutated[j, i] = conc_ij/(row_sum)
    conc_offspring_mutated = mutation2(conc_offspring_mutated,
                                       mutation_rate_2)
    conc_offspring_mutated = np.abs(conc_offspring_mutated)
    for j in range(conc_offspring_mutated.shape[0]):
        row_sum = np.sum(conc_offspring_mutated[j, :])
        for i in range(conc_offspring_mutated.shape[1]):
            conc_ij = conc_offspring_mutated[j, i]
            conc_offspring_mutated[j, i] = conc_ij/(row_sum)
    return conc_offspring_mutated, median_fitness, max_fitness


def GA_algorithm_unnormalized(x_train_spectra, y_train_conc,
                              x_test, n_parents, n_offspring,
                              mutation_rate, mutation_rate_2):
    np.random.seed(seed)
    # Obtains fitness of the input concentrations and their spectra
    array, median_fitness, max_fitness = fitness(x_train_spectra,
                                                 y_train_conc, x_test)
    # Obtains the parents of the ordered fitness array
    parents = select_parents(array, n_parents)
    # Prepares the parents array to remove the spectra information
    parents = parents[:, parents.shape[1] - y_train_conc.shape[1]-1:]
    # Performs a crossover to obtain the offspring of the parents
    offspring = crossover(parents, n_offspring)
    # conc_offspring = offspring[:,x_train_spectra.shape[1]:-1]
    conc_offspring = offspring
    conc_offspring_unique = conc_offspring
    # conc_offspring_unique = np.unique(conc_offspring, axis=0)
    # Perfroms a mutation on the offspring array
    conc_offspring_mutated = mutation(conc_offspring_unique, mutation_rate)
    # Filters offspring array for negative values and normalizes all values
    conc_offspring_mutated = np.abs(conc_offspring_mutated)
    return conc_offspring_mutated, median_fitness, max_fitness


def perform_iteration(current_gen_spectra, current_gen_conc, desired_spectra,
                      n_parents, n_offspring, mutation_rate, mutation_rate_2):
    ''' Perfroms one iteration of the GA algorithm.
    Inputs:
    - current_gen_spectra: The spectra of the current generation (batch).
    It is a 2D array with the number of rows equal to the number of samples
    in the generation and number of colums equal to the number of spectra
    datapoints.
    - current_gen_conc: The concentration of the current generation (batch).
    It is a 2D array with the number of rows equal to the number of samples
    in the generation and the number of columns equal to the number of
    dimensions, for exmaple, 3 columns if we are mixing red, blue, green dyes.
    - desired_spectra: The desired spectra. It is a 1D array with one row
    and number of columns equal to the number of datapoints in the spectra.
    - n_parents: Integer which determines how many parents to create from the
    current generation.
    - n_offspring: Integer which determines how many offspring to create
    from the current generation.
    - mutation_rate: Float from range 0-1 which determines how often a
    mutation occurs.
    - mutation_rate_2: Float from range 0-1 which deterines how often a
    mutation occurs.
    Outputs:
    - next_gen_conc: The concentrations of the next generation to be tested.
    It is a 2D array with number of rows equal to n_offspring and number of
    columns equal to the number of dimensions. '''
    np.random.seed(seed)
    cgs = current_gen_spectra.T
    current_gen_spectra = mms().fit(cgs).transform(cgs).T
    desired_spectra = prepare_desired_spectra(desired_spectra)
    # Perfrom Genetic Algorithm to determine next Generation
    next_gen_conc, median_fitness, max_fitness = GA_algorithm(
      current_gen_spectra,
      current_gen_conc,
      desired_spectra,
      n_parents, n_offspring,
      mutation_rate,
      mutation_rate_2)
    return next_gen_conc, median_fitness, max_fitness
