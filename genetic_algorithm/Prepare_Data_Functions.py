import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def load_df(directory):
    df = pd.read_excel(directory)
    return df 
    
def subtract_baseline(df, baseline):
    x = df.values
    cols = df.columns
    baseline_col = df[baseline].values
    for i in range(1,x.shape[1]):
        x[:,i] = x[:,i] - baseline_col
    df = pd.DataFrame(x, columns = cols)
    return df 

def delete_rows(df, rows_delete):
    x = df.values
    cols = df.columns
    x = x[rows_delete:,:]
    df = pd.DataFrame(x, columns = cols)
    return df
    
def normalize_df(df):
    x = df.values #returns a numpy array
    cols = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = np.hstack((x[:,0].reshape(-1,1),x_scaled[:,1:]))
    df = pd.DataFrame(x_scaled, columns = cols)
    return df 

def plot_single_graph(df, column):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df['Wavelength'], df[column])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')
    
def find_max_wavelength(df, column):
    array = np.vstack((df['Wavelength'], df[column])).T
    sorted_array = array[np.argsort(array[:, 1])]
    max_wavelength = sorted_array[-1,0]
    return max_wavelength

def plot_all_spectra_multiple(df):    
    for i in range(1, len(df.columns)):
        plot_single_graph(df.columns[i])

def plot_all_spectra_single(df):    
    for i in range(1, len(df.columns)):
        plt.plot(df['Wavelength'], df[df.columns[i]])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
            
def plot_some_spectra_single(df, cols):    
    plt.figure(figsize=(8,5))
    for i in range(0, len(cols)):
        plt.plot(df['Wavelength'], df[cols[i]]) 
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.xlim([400,800])
        # plt.legend()
        
        