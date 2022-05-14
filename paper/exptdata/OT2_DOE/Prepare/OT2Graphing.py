import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rearrange(sample_volumes):
    """Rearranges sample information to group samples based on position in sublist. [[a1,b1,c1],[a2,b2,c2]] => [[a1,a2],[b1,b2],[c1,c2]]"""
    component_volumes_rearranged = []
    for i in range(len(sample_volumes[0])): 
        component_volumes = []
        for sample in sample_volumes:
            component_volume = sample[i]
            component_volumes.append(component_volume)
        component_volumes_rearranged.append(component_volumes)
    return component_volumes_rearranged 

def graph_canidates(experiment_csv_dict, unfiltered, filtered, additional_text = None, xlim = [0,1], ylim = [0,1]):
    experiment_csv_dict = experiment_csv_dict
    unfiltered_samples = unfiltered
    filtered_samples = filtered
    
    wtf_unfiltered_rearranged = rearrange(unfiltered_samples)
    wtf_filtered_rearranged = rearrange(filtered_samples)

    marker_size = 20

    component_names = experiment_csv_dict['Component Shorthand Names']
    
    x_index = experiment_csv_dict['Component Graphing X Index']
    x_component_name = component_names[x_index]

    y_index = experiment_csv_dict['Component Graphing Y Index']
    y_component_name = component_names[y_index]
    

    plt.scatter(wtf_unfiltered_rearranged[x_index], 
                wtf_unfiltered_rearranged[y_index], 
                marker_size, alpha = 0.6, marker = 'x', color = 'b')

    # Plot samples actually made 
    plt.scatter(wtf_filtered_rearranged[x_index], # could automate by looking at the experiment names
                wtf_filtered_rearranged[y_index], 
                marker_size, alpha = 0.5, marker = 'o', color = 'r')

#     plt.plot([0,1],[0,0.02])
    
    plt.xlabel(x_component_name + ' wtf')
    plt.ylabel(y_component_name + ' wtf')
    
    completing_component_index = len(component_names) - 1  
    other_indexes = []
    for i in range(len(component_names)):
        if i == x_index or i == y_index or i == completing_component_index:
            pass
        else:
            other_indexes.append(i)
        
    component_units = experiment_csv_dict['Component Concentration Unit']
    component_concentrations = experiment_csv_dict['Component Concentrations [min, max, n]']
    
    text = []
    for index in other_indexes: 
        component_unit = component_units[index]
        component_concentration = component_concentrations[index][0]
        component_name = component_names[index]
        string = component_name + ' ' + component_unit + ' ' + str(component_concentration)
        text.append(string)
    
    text.append('Remaining component = ' + component_names[completing_component_index])
    
    if additional_text is not None:
        text = text + additional_text
            
    text_newline = '\n'.join(text) 
        

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.annotate(text_newline, xy=(1.05, 0.5), xycoords='axes fraction')        
    plt.autoscale(True)
    plt.show()

def stock_search(experiment_info_dict, unfiltered_wtfs, stock_canidates_samples, stock_text_list): 
    wtf_sample_canidates = experiment_info_dict['wtf_sample_canidates']
    for i, stock_canidate_sample in enumerate(stock_canidates_samples):
        stock_text_list[i].append('Index = ' + str(i))
        graph_canidates(experiment_info_dict, wtf_sample_canidates, stock_canidate_sample, additional_text = stock_text_list[i])
        
def stock_search_sep(stock_dict, experiment_csv_dict, unfiltered_wtfs): 
    stock_canidates_samples = stock_dict['stocks_wtf_lists']
    stock_text_list = stock_dict['stock_text_info']

    for i, stock_canidate_sample in enumerate(stock_canidates_samples):
        stock_text_list[i].append('Index = ' + str(i))
        graph_canidates(experiment_csv_dict, unfiltered_wtfs, stock_canidate_sample, additional_text = stock_text_list[i])
        
def baseline_correction(df_samples, baseline_series): 
    """Given the series iloc of a the blank, subtracts the value at every wavelength of blank at resp. wavelength. 
    Simple subtraction blanking."""
    new_df_con = []
    for key, row in df_samples.iterrows():
        if key == 'Wavelength':
            wavelengths = row
            new_df_con.append(wavelengths)
        else: 
            series = row
#             series = (pd.to_numeric(series , errors='coerce').fillna(0)) # just know that the series is an instance so will not update df
            corrected = series.subtract(baseline_series)
            new_df_con.append(corrected)
    
    baseline_corrected_df = pd.concat(new_df_con, axis = 1).T
    baseline_corrected_df.index = df_samples[0].index
    return baseline_corrected_df

def plot_single_wavelength(dataframe, wavelength):
    for i, (key, row) in enumerate(dataframe.iterrows()):
        if key == 'Wavelength':
            wavelengths = row
    
    index = np.where(wavelengths == wavelength)[0][0]
    wells = []
    absorbances = []
    for i, (key, row) in enumerate(dataframe.iterrows()):
        if key == 'Wavelength':
            pass
        else:
            well = key
            wells.append(well)
            absorbance = row[index]
            absorbances.append(absorbance)

    plt.scatter(range(len(wells)), absorbances, s = 20, alpha = 0.5, marker = 'o', color = 'r') # in order for sample creation and analysis 
    plt.xlabel('Well Index')
    plt.ylabel('Absorbance')
    
    return absorbances

def plot_wavelengths_zoom(dataframe, labels, x_lim = [], y_lim = []):
    """Given a dataframe with the wavelegnth information as one row will plot and will zoom based on provided axis"""
    fig, ax = plt.subplots()
    for i, (key, row) in enumerate(dataframe.iterrows()):
        if key == 'Wavelength':
            x = row
        else: 
            y = row
            ax.plot(x,y,label = labels[i])
    ax.legend()
    
    plt.xlabel('Wavelength nm')
    if len(x_lim) == 2:
        plt.xlim(x_lim)
    
    plt.ylabel('Absorbance')
    if len(y_lim) == 2:
        plt.ylim(y_lim)


def xy_scatter_df(df, x_name, y_name):
    """Plots two columns of a dataframe against each other on a 2D scatter plot"""
    x = df[x_name]
    y = df[y_name]
    
    fig, ax = plt.subplots(1)
    ax.scatter(x, y, marker="+", c = 'black')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    
def xy_scatter_df_compare(df1, df2, x_name, y_name):
    """Plots two columns of a 2 dataframe against each other on a 2D scatter plot. Allows for a visual comparasion."""
    x1 = df1[x_name]
    y1 = df1[y_name]
    
    x2 = df2[x_name]
    y2 = df2[y_name]
    
    fig, ax = plt.subplots(1)
    ax.scatter(x1, y1, marker="s", alpha=0.4, c = 'orange')
    ax.scatter(x2, y2, marker="+", c = 'black')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)