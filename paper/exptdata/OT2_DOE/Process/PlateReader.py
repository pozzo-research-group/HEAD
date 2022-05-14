import pandas as pd
import numpy as np


def merge_wavelength_dfs(df_list):
    merge_list = []
    for i, df in enumerate(df_list):
        if i == 0:
            df = df
        else: 
            df = df.drop(['Wavelength'])
        merge_list.append(df)
    return pd.concat(merge_list)

def detect_ovflw(df, holder=15):
    hold_index = df.index
    
    df = df.copy() # Setting up to prevent any errors
    df.reset_index(drop=True, inplace=True)

    for name, row in df.iterrows():
        row = list(row)
        if 'OVRFLW' in row: 
            row = [holder if value == 'OVRFLW' else value for value in row]
            df.loc[name] = row
        else:
            df.loc[name] = row
    df.index = hold_index
    
    return df

def extract_plates(path, sheet_list):
    plate_dfs = []
    for sheet_name in sheet_list:
        plate_df = pd.read_excel(path, sheet_name = sheet_name).T
        plate_dfs.append(plate_df)
    return plate_dfs

def add_abs_to_sample_info(sample_info_df, abs_df):
    
    wavelengths = list(abs_df.loc['Wavelength'])
    wavelengths_names = [str(wavelength)+'nm' for wavelength in wavelengths]
    abs_df.columns = wavelengths_names
    abs_df.drop(['Wavelength'], inplace=True)

    
    
    sample_info_df.reset_index(drop=True, inplace=True)
    abs_df.reset_index(drop=True, inplace=True)
    combined_df = pd.concat([sample_info_df, abs_df], axis = 1)
    return combined_df

def rehead_wavelengths(platereader_df, add_unit = 'nm'):
    
    platereader_df = platereader_df.copy() # because dataframes reference past definition, especially important when deleting/modifying things
    wavelengths = list(platereader_df.loc['Wavelength'])
    wavelengths_names = [str(wavelength)+'nm' for wavelength in wavelengths]
    
    platereader_df.columns = wavelengths_names
    platereader_df.drop(['Wavelength'], inplace = True)
    
    return platereader_df