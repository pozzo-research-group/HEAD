import numpy as np
import pandas as pd
import os
import csv
import ast
import datetime
from pytz import timezone

mass_dictionary = {'g':1} # should build a dictionary of units people can add to such that not restricted to hardcoded ones


##### Set up the experiment plan dictionary to be referenced for useful information throughout a design of experiments. This is not necessary if loading in volumes directly#####

def get_experiment_plan(filepath, chemical_database_path):
    """
    Parse a .csv file to create a dictionary of instructions.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        plan_dict = {}
        for i, row in enumerate(reader):
            assert len(row) == 2
            plan_dict[row[0]] = ast.literal_eval(row[1])
    
    chem_data = pd.read_csv(chemical_database_path)
    chem_data_names = chem_data['Component Abbreviation']
    chem_data.index = chem_data_names
    plan_dict['Chemical Database'] = chem_data.T.to_dict()
    
    return plan_dict

def component_order_dictionary(plan):
    """Would hold a nested dictionary for each component for the case of maintaining the order and not having to repeat the calling 
    of list, this will make it less prone to errors when looking at names, units and linspace."""
    component_order_dict = {}
    for key, value in plan.items():
        if 'Component' in key:
            component_order_dict[key] = value

    return component_order_dict


##### Create the concentrations dataframe. Dataframe must be structured where the column names must be component name followed by the word concentration and the selected unit (i.e. Cadmium component mgpermL) ####
##### NOTE: Concentration can refer to either the actual molecular components, but if using volf then it can refer to the stocks themselves. In this case of components = stocks, the only other information needed to caculate volume is total volume of each stock.####
##### You can also use these functions to import volume plans if you do not wish to use any of the built in caclulation modules which follow #####

def concentration_from_csv(csv_path):
    """Given a path to a csv will translate to the information to a dataframe in the default nature of pandas.
    Data is formatted based on column and spacing, hence in a csv the first row will headers seperated by commas and the next row will 
    be respective header values seperated by commas."""
    concentration_df = pd.read_csv(csv_path)
    concentration_df = concentration_df.astype(float)
    return concentration_df 

def concentration_from_excel(excel_path):
    """Given a path to an excel file (xlsx) will translate information to a dataframe in the default nature of pandas.
    Data is formatted based on the same row and column order within the excel sheet.
    Ensure the headers match the names of components of a plan as information
    """
    concentration_df = pd.read_excel(excel_path)
    concentration_df = concentration_df.astype(float)
    return concentration_df

def concentration_from_linspace(component_names, component_linspaces, component_units, unity_filter = False, component_spacing_type='linear'):
    """ Uses linspaces to create a mesh of component concentrations. The linspaces are pulled from a csv plan where all arguments are in parellel (i.e position 1 of arugment 1 refers to position 1 of argument 2). 
    Hence linspaces pulled from plan are requried to match with component names and component unit arguments. The only exception is that you may leave the last linspace unspecified if you looking to complete the argument 
    using a unity filter. The application of the unity filter looks at the last position in the component names list and uses the other calculated component concentration values to find the difference of 1 to complete a sample concentration values. 
    """

    conc_range_list = [] 
    for conc_linspace in component_linspaces:
        if component_spacing_type == "linear": 
            conc_range_list.append(np.linspace(*conc_linspace))
        elif component_spacing_type == "log": 
            conc_range_list.append(np.logspace(*conc_linspace))
    conc_grid = np.meshgrid(*conc_range_list)

    component_conc_dict = {} 
    for i in range(len(conc_grid)): 
        component_name = component_names[i]
        component_unit = component_units[i]
        component_values = conc_grid[i].ravel()
        component_conc_dict[component_name + " " + 'concentration' + " " + component_unit] = conc_grid[i].ravel()
    concentration_df = pd.DataFrame.from_dict(component_conc_dict)

    # this is only here for linspaces as why? it should just be an option for all of them
    if unity_filter == True: # POTENTIAL ISSUE IF LEFT ON AS TRUE EVEN IF NOT USING COMPLETING FORMULATION CAN OVERWRITE
        unity_filter_df(concentration_df, component_names, component_units)

    return concentration_df

def concentration_from_list_componentwise_grid(component_names, component_concentrations_sublist, component_units, unity_filter = False, component_spacing_type='linear'):
    """ Given the component names, units and concentrations in parallel will create a concentration dataframe. The concentration values are to formatted where each sublist contains all the information for a single SAMPLE
    matching the order of component names and units. Each concentration from each component is used in combination with each other component concentration to create a combination of samples. For example component names = [comp1, comp2, comp3] then concentration_sublists = [[comp1_sample1, comp2_sample1, comp3_sample1], [comp1_sample2, comp2_sample2, comp3_sample2]]
    """

    conc_range_list = [] 
    for conc_space in component_concentrations_sublist:
        conc_range_list.append(conc_space)
    conc_grid = np.meshgrid(*conc_range_list)

    component_conc_dict = {} 
    for i in range(len(conc_grid)): 
        component_name = component_names[i]
        component_unit = component_units[i]
        component_values = conc_grid[i].ravel()
        component_conc_dict[component_name + " " + 'concentration' + " " + component_unit] = conc_grid[i].ravel()
    concentration_df = pd.DataFrame.from_dict(component_conc_dict)

    if unity_filter == True:
        unity_filter_df(concentration_df, component_names, component_units)

    return concentration_df

def concentration_from_list_samplewise(component_names, concentration_sublists, component_units):
    """ Given the component names, units and concentrations in parallel will create a concentration dataframe. The concentration values are to formatted where each sublist contains all the information for a single SAMPLE
    matching the order of component names and units. For example component names = [comp1, comp2, comp3] then concentration_sublists = [[comp1_sample1, comp2_sample1, comp3_sample1], [comp1_sample2, comp2_sample2, comp3_sample2]]
    """
    same_len(concentration_sublists)
    assert len(component_names) == len(next(iter(concentration_sublists))), 'Length of concentration sublists is not in line with component names'
    assert len(component_names) == len(component_units), 'Length of component units is not in line with component names'

    column_names = [name + ' ' + unit for name, unit in zip(component_names, component_units)]
    concentration_df = pd.DataFrame(data=concentration_sublists)
    concentration_df.columns = column_names

    return concentration_df

def concentration_from_list_componentwise(component_names, concentration_sublists, component_units):
    """ Given the component names, units and concentrations in parallel will create a concentration dataframe. The concentration values are to formatted where each sublist contains all the information for a single COMPONENT
    matching the order of component names and units. For example component names = [comp1, comp2, comp3] then concentration_sublists = [[comp1_sample1, comp1_sample2, comp3_sample1], [comp1_sample2, comp2_sample2, comp3_sample2]]
    """
    same_len(concentration_sublists)
    assert len(component_names) == len(next(iter(concentration_sublists))), 'Length of concentration sublists is not in line with component names'
    assert len(component_names) == len(component_units), 'Length of component units is not in line with component names'

    column_names = [name + ' ' + unit for name, unit in zip(component_names, component_units)]
    concentration_df = pd.DataFrame(data=concentration_sublists).T
    concentration_df.columns = column_names

    return concentration_df

def unity_filter_df(concentration_df, component_names, component_units):
    """For units which sum to one, will create an additional column to represent the final component. This will require 
    that the input information such as sample names have this completing component as the last entry. Currently no general way 
    to verify if sample is under of overspecified, must verify yourself.
    """

    completing_index = len(component_names)-1
    completing_component_name = component_names[completing_index]
    completing_component_unit = component_units[completing_index]
    completing_entry_name = completing_component_name + " " + 'concentration' + " " + completing_component_unit
    concentration_df[completing_entry_name] = (1 - concentration_df.sum(axis=1)) 
        
    unfiltered_concentration_df = concentration_df # used to catch errors when concentration_df after fully defined concentration produces no suitable canidates
    
    concentration_df = concentration_df[concentration_df[completing_entry_name] > 0]
    concentration_df.reset_index(drop=True, inplace=True)

    assert not concentration_df.empty, 'No suitable samples were found, please change your concentration space. Most likely this means you have your linspaces set too close together at all high concentrations (close to 1) resulting in impossible samples (wtf/volf>1). Turn on expose_df to return unfiltered dataframe'
    return concentration_df


#### CALCULATIONS FUNCTIONS: Once you have created the dataframe of desired concentration (either refering to concs of molecular species or stocks (restricted to only volf)) the following functions will help you calculate respective volumes #####
#### If caclulating from concentrations then you will need to provide the following information: total sample amount, total sample unit, stock_names, stock_concentrations, stock_unit. Component information will be pulled from the columns names. 

def determine_component_mass(total_sample_amount, total_sample_amount_unit, component_values, component_unit, component_info):
    """Determines the mass of a component (series or single value) based on the total sample unit and the component unit. 
    Hence there are a finate number of unit combinations which will work, the easiest way to think about this look the numerator 
    and denominator of the component unit and determine how to remove the denominator."""
    
    if total_sample_amount_unit == 'g' and component_unit == 'wtf':
        component_masses = total_sample_amount*component_values

    elif total_sample_amount_unit == 'mL' and component_unit == 'mgpermL':
        component_masses = total_sample_amount*component_values/1000 # for now default mass = g, volume = mL

    elif total_sample_amount_unit == 'mL' and component_unit == 'molarity':
        molecular_weight = component_info['Molecular Weight (g/mol)']
        component_masses = total_sample_amount*component_values*molecular_weight/1000
    
    else: 
        raise AssertionError(total_sample_amount_unit, 'and', component_unit, 'units are not supported to calculate for mass')
    print('You calculated for component masses given the provided units')
    return component_masses

def determine_component_volumes(total_sample_amount, total_sample_amount_unit, component_values, component_unit, component_info):
    """Determines the volume of a component (series or single value) based on the total sample unit and the component unit. 
    Hence there are a finate number of unit combinations which will work, the easiest way to think about this look the numerator 
    and denominator of the component unit and determine how to remove the denominator."""
    
    if total_sample_amount_unit == 'mL' and component_unit == 'volf':
        component_volume = total_sample_amount*component_values
        print('You calculated for component volumes given the provided units')
        return component_volume

def determine_component_amounts(plan, concentration_df, nan_fill_value = None):
    """Based on plan information (Component Names and total sample unit) will determine the amount of each component (mass and volume) 
    required for each sample. Currently only supports mL and g as default units as this is the density unit basis pulled from a the chemical database.
    It is recommended you keep plan lists in order i.e component[2] refers to the third column of components. Ensure you do not modify the column names."""
    concentration_df = concentration_df.copy()
    component_info_dict = plan['Chemical Database']
    total_sample_amount_unit = plan['Sample Unit']
    total_sample_amount = plan['Sample Amount']
    
    for column_name in concentration_df: 
        component_name = identify_component_name(column_name)
        component_unit = identify_unit(column_name)
        component_values = concentration_df[column_name]
        component_info = component_info_dict[component_name]
        component_density = component_info['Density (g/mL)']
        if component_unit == 'wtf' or 'mgpermL' or 'molarity': # these are the unit that lead to mass outcomes
            component_masses = determine_component_mass(total_sample_amount, total_sample_amount_unit, component_values, component_unit, component_info)  
            component_volumes = component_masses/component_density
            concentration_df[component_name + ' amount mass ' + 'g']  = component_masses
            concentration_df[component_name + ' amount volume ' + 'mL']  = component_volumes
        if component_unit == 'volf': # these are the unit that lead to volume outcomes
            component_volumes = determine_component_volumes(total_sample_amount, total_sample_amount_unit, component_values, component_unit, component_info)  
            component_masses = component_volumes*component_density
            concentration_df[component_name + ' amount volume ' + 'mL']  = component_volumes
            concentration_df[component_name + ' amount mass ' + 'g']  = component_masses
    
    if nan_fill_value is not None:
        concentration_df = concentration_df.fillna(nan_fill_value)

    return concentration_df

def stock_dictionary(stock_names, stock_units, stock_values, stock_densities = None):
    """Creating a dictionary which will contain information tied to a stocks name. The arugments provides must be in list form with positions being in parallel. 
    The stock name is required to be in the form 'solute1...-soluten-solvent-stock' where the entry prior to the keyword stock are solvent and anything prior to that is assumed a solute.
    Stock density is an optional argument as it is only necessary for one pathway (when using stock wtf), if only some stocks are known ensure the unknown are placed as nan. In the case of a stock 
    being purely one component with known density do this [nan, known density, nan] ... etc"""

    if stock_densities == None:
        stock_densities = len(stock_names)*[float('nan')]
    
    stock_dict = {}
    for i, stock_name in enumerate(stock_names):
        stock_unit = stock_units[i]
        stock_value = stock_values[i]
        stock_density = stock_densities[i]
        stock_components = stock_name.split('-')
        stock_solutes = stock_components[:-2] # will always be a list
        stock_solvent = stock_components[-2]
        stock_dict[stock_name] = {'solutes': stock_solutes, 'solvents':stock_solvent, 'unit': stock_unit, 'concentration': stock_value, 'Density (g/mL)': stock_density}
    
    stock_dict = identify_common_solvents(stock_dict)
    return stock_dict

def identify_common_solvents(stock_dict):
    """ Given the stock_dict will identify solvents which are present in more than one stock. It will add an entry to each dictionary of stock information with the key = 'Common Solvent':value = 
    None = Stock has no shared solvents 
    Mixed = Stock has shared solvents AND at least one solute
    Pure = Stock is a shared solvent (now identifiable to complete mixtures and account if looking for specfic common solvent concentration)
    """
    solvents = [stock_info['solvents'] for stock_name, stock_info in stock_dict.items()]
    common_solvents = list(set([x for x in solvents if solvents.count(x) > 1]))
    for common_solvent in common_solvents: 
        for stock_name, stock_info in stock_dict.items():
            stock_solutes = stock_info['solutes']
            stock_solvents = stock_info['solvents']
            if len(stock_solutes) == 0 and common_solvent== stock_solvents:
                stock_dict[stock_name]['Common Solvent'] = 'Pure'
            elif len(stock_solutes) > 0 and common_solvent == stock_solvents:
                stock_dict[stock_name]['Common Solvent'] = 'Mixture'
            elif stock_dict[stock_name].get('Common Solvent'): # prevents from Pure and Mixtures being overwriteen by None
                pass
            else:     
                stock_dict[stock_name]['Common Solvent'] = 'None'

            
    return stock_dict


def calculate_stock_volumes_mass_units(component_mass, component_unit, stock_concentration, stock_unit, stock_density = None, component_mw = None):
    """ Based on the most base information of: 
    - Component mass and unit
    - Stock concentration anf unit 
    - Optional stock and component info
    will calculate and return the volume of stock needed to achieve the provided mass of component. Currently hardcoded units are only supported, grams is the only mass. 
    """

    # hmm maybe add something to catch if someone forgets or has density or mw as nan. 
    if component_unit == 'g' and stock_unit == 'wtf': # we need to specify g as it is the only thing working
        stock_mass = component_mass/stock_concentration # in g
        stock_volume = stock_mass/stock_density
    elif component_unit == 'g' and stock_unit == 'molarity':
         stock_volume = 1000*component_mass/(stock_concentration*component_mw) # in mL
    elif component_unit == 'g' and stock_unit == 'mgpermL':
         stock_volume = component_mass/(stock_concentration/1000)
    else:
        raise AssertionError("Units provided are not currently supported for component mass to stock volume calculations")
    return stock_volume

def calculate_stock_volumes_vol_units(component_volume, component_unit, stock_concentration, stock_unit, stock_density = None, stock_mw = None):
    # mL could really be anything
    if component_unit == 'mL' and stock_unit == 'volf': # we need to specify g as it is the only thing working
        stock_volume = component_volume/stock_concentration

    else: 
        raise AssertionError("Units provided are not currently supported for component volume to stock volume calculations")
    return stock_volume


def calculate_stock_volumes_from_component_concs(plan, complete_component_df, stock_dict): # this is working to be more automatic

    component_dict = plan['Chemical Database']


    component_concentrations = isolate_common_column(complete_component_df, 'concentration')
    component_masses = isolate_common_column(complete_component_df, 'mass')
    component_volumes = isolate_common_column(complete_component_df, 'volume')
    for stock_name, stock_info in stock_dict.items():
        stock_unit = stock_info['unit']
        stock_concentration = stock_info['concentration']
        stock_density = stock_info['Density (g/mL)']
        
        if len(stock_info['solutes']) != 0:
            component_name = stock_info['solutes'][0]
        else:
            component_name = stock_info['solvents']
        
        
        concentration_column = find_component_column(component_name, component_concentrations)
        component_conc_unit = identify_unit(concentration_column[0])
#         component_concs = component_concentrations[concentration_column]
        # ok using the concentraiton you determined the path to take and which information to pull
        
        if component_conc_unit in ('wtf', 'molarity', 'mgpermL') and stock_unit in ('wtf', 'molarity', 'mgpermL'):
            component_mass_column = find_component_column(component_name, component_masses)
            component_unit = identify_unit(component_mass_column[0])
            component_mass = component_masses[component_mass_column]
            component_mw = component_dict[component_name]['Molecular Weight (g/mol)']
            stock_volumes = calculate_stock_volumes_mass_units(component_mass, component_unit, stock_concentration, stock_unit, stock_density, component_mw)
            
        if component_conc_unit in ('volf', 'molarity', 'mgpermL') and stock_unit in ('volf'): # hmm this is odd, but essentially if you see volf with it is just easier to use volumes as the basis over mass, but be careful since you can calualte
            #so leave both the option to use all masses or if everything has defined density then okay to use, this was more made for pure liquids
            component_volume_column = find_component_column(component_name, component_volumes)
            component_unit = identify_unit(component_volume_column[0])
            component_volume = component_volumes[component_volume_column]
            
            stock_volumes = calculate_stock_volumes_vol_units(component_volume, component_unit, stock_concentration, stock_unit)
        
        complete_component_df[stock_name + ' amount volume mL'] = stock_volumes
    return complete_component_df


def calculate_stock_volumes_from_component_masses(plan, complete_component_df, stock_dict): # this can be trouble some since it restirct you from ever mixing volf and the other units, it makes all basis of mass, what instead should be done is basedon the unit of both the stock and component it should direct to appropiate function
    """Used to calculate stock volume from component volumes. This pathway is only appropiate when dealing with component masses and stock units of wtf, molarity, and mgpermL.
    This is still under the assumption of one component + one solvent = one stock and that the complete_component_df headers will be in the form componentname_rest of column.
    """

    component_dict = plan['Chemical Database']
    component_masses = isolate_common_column(complete_component_df, 'mass')
    for stock_name, stock_info in stock_dict.items():
        stock_unit = stock_info['unit']
        stock_concentration = stock_info['concentration']
        stock_density = stock_info['Density (g/mL)']
        
        if len(stock_info['solutes']) != 0:
            component_name = stock_info['solutes'][0]
        else:
            component_name = stock_info['solvents']
        
        component_mass_column = find_component_column(component_name, component_masses)
        component_unit = identify_unit(component_mass_column[0])
        component_mass = component_masses[component_mass_column]
        component_mw = component_dict[component_name]['Molecular Weight (g/mol)']
        
        stock_volumes = calculate_stock_volumes_mass_units(component_mass, component_unit, stock_concentration, stock_unit, stock_density, component_mw)
        complete_component_df[stock_name + ' amount volume mL'] = stock_volumes
    return complete_component_df


#### If common solvents are present then use these functions to account for them. Each has its specific use case so understand the information you need

def missing_volume(total_sample_volume, complete_df):
    """Simple calculation to compute for a missing volume. Reccomended you use complete_missing_volume_with_commmon_solvent() function."""
    stock_df = isolate_common_column(complete_df, 'stock')
    total_stock_volume = stock_df.sum(axis=1)
    missing_volume = total_sample_volume-total_stock_volume
    complete_df['Missing Sample Volume mL'] = missing_volume
    return complete_df

def complete_missing_volume_with_commmon_solvent(complete_volume, complete_df, stock_dict, solvent=None):
    """Aimed to caclulate for missing solvent volumes (based on the complete volume argument) when working with typically mgpermL or molarity component based DOE.
    Will first look if a single common solvent is present by referencing the stock dictionary made previous, if there is none
    or more than two common solvents then it is required to manually enter it as an argument."""
    complete_df = complete_df.copy()
    stock_volumes = isolate_common_column(complete_df, 'stock')
    stock_names = [identify_component_name(col) for col in stock_volumes if 'stock' in col] # this is seperate from the dict since your dict can have more stocks just need ot have the same name
    pure_common_solvent_stocks = [stock_name for stock_name in stock_names if stock_dict[stock_name]['Common Solvent'] == 'Pure']
    mixture_common_solvent_stocks = [stock_name for stock_name in stock_names if stock_dict[stock_name]['Common Solvent'] == 'Mixture']

    assert not len(pure_common_solvent_stocks) > 1, 'Too many common solvents, select one by specifying solvent arugment'
    if solvent:
        solvent = solvent
    elif len(pure_common_solvent_stocks) == 1:
        solvent = next(iter(pure_common_solvent_stocks))
    else:
        raise AssertionError('Solvent has not been selected, either specify one or ensure common solvent already presnent in stock dictionary')
    
    missing_solvent_volume = complete_volume-stock_volumes.sum(axis=1)
    complete_df[solvent + '-stock volume mL'] = missing_solvent_volume
    return complete_df

def calculate_common_solvent_residual_volumes(complete_df, stock_dict):
    """ By looking at common solvent arguments previously established in the stock_dict, will take into account stock volumes which contain a common solvent and if the commmon solvent is 
    present as a stock it will subtract the volume of common solvent from it leaving you the appropiate common solvent volume. 

    Need to modify or make own function as if there is common solvents but no common solvent stock that it will make one to complete the volume if needed. 
    """
    complete_df = complete_df.copy() # why is this necessary
    stock_volumes = isolate_common_column(complete_df, 'stock')
    
    stock_names = [identify_component_name(col) for col in stock_volumes if 'stock' in col] # this is seperate from the dict since your dict can have more stocks than present in the df
    
    pure_common_solvent_stocks = [stock_name for stock_name in stock_names if stock_dict[stock_name]['Common Solvent'] == 'Pure']
    mixture_common_solvent_stocks = [stock_name for stock_name in stock_names if stock_dict[stock_name]['Common Solvent'] == 'Mixture']

    for pure_common_solvent in pure_common_solvent_stocks:
        for stock_name in mixture_common_solvent_stocks:
            stock_info = stock_dict[stock_name]
            solutes = stock_info['solutes']
            if pure_common_solvent in stock_name:
                stock_volumes_to_subtract = find_best_df_match(complete_df, stock_name + ' amount volume') 
                component_volumes = find_best_df_match(complete_df, solutes[0] + ' amount volume') 
                stock_volumes_to_subtracted_solvent = stock_volumes_to_subtract-component_volumes.values
                # need to somehow get rid of the mL dependence, the easiest way is to make the stock_info prior hold its volume headers as to call easier
                common_stock_volume_to_track_name = [pure_common_solvent + ' amount volume mL']
                common_stock_volume_to_track = complete_df[common_stock_volume_to_track_name] 
                common_stock_volume_removed_stocks = common_stock_volume_to_track - stock_volumes_to_subtracted_solvent.values
                complete_df[common_stock_volume_to_track_name] = common_stock_volume_removed_stocks
    
    return complete_df 

def calculate_common_solvent_missing(): 
    # well this could just be an argument of the function above by seeing if the column exist and if it does continue and if not make one based on the pure_common_solvent list 
   
    pass

def add_final_location(directions, complete_df, unique_identifier= None, date_MM_DD_YY=None):    
    complete_df = complete_df.copy()
    info = []
    for i, sample_info in directions.items():
        for stock, variable in sample_info.items():
            final_well_destination = variable['Destination Well Position']
        info.append(final_well_destination)    
    
    if date_MM_DD_YY is not None:
        time = date_MM_DD_YY
    else:
        time = datetime.datetime.today().strftime('%m-%d-%Y') # str(datetime.datetime.now(timezone('US/Pacific')).date()) # should be embaded once you run

    wells = []
    labwares = []
    slots = []
    info_cut = info #info only being used of length of number of samples
    for info in info_cut:
        # string consist of three components, well_of_labware__on_slot with of and on being the seperators which is native and consistent across all OT2 protocols
        string = str(info)
        lower_seperator = 'of'
        upper_seperator = 'on'

        lower_seperator_index = string.index(lower_seperator)
        upper_seperator_index = string.rindex(upper_seperator)
        well = string[:lower_seperator_index-1]
        labware = string[lower_seperator_index + len(lower_seperator)+ 1:upper_seperator_index-1]
        slot = string[upper_seperator_index+len(upper_seperator)+1:]

        wells.append(well)
        labwares.append(labware)
        slots.append(slot)

    UIDs = []
    for slot, labware, well in zip(slots, labwares, wells):
        UID = "S" + slot + "_" + well + "_" + time  # add name of interest here to make it easier to identify
        if unique_identifier is not None: 
            UID = UID + "_" + str(unique_identifier)
        UIDs.append(UID)

    complete_df.insert(0, 'UID', UIDs)
    complete_df.insert(1, 'Labware', labwares)
    complete_df.insert(2, 'Slot', slots)
    complete_df.insert(3, 'Well', wells)
    return complete_df

def create_labels_for_plate():
    pass

def create_labels_for_wells():
    pass

#### These are utility function no single use 

def identify_unit(string):
    """Based on a provided string will identify if a unit within a list verified working units"""
    supported_units = ['wtf','volf','molf','mgpermL','molarity', 'g', 'mL', 'g/mL', 'g/mol']
    for unit in supported_units:
        if unit in string:            
            return unit
    raise AssertionError('Unit in ' + string + ' not currently supported, the following units are supported: ', supported_units)

def identify_component_name(string):
    """Will pull the first word from the string and return it to be used as the name to look up in a chemical database.
    Can also make this is a checkpoint to ensure the component is in the chemical database to begin with. This allows you 
    to contain all information within a column (component identity, name)"""
    component = string.split(' ', 1)[0]
    return component

def same_len(iterable_2d):
    """Checks if all nested iterables are the same length."""
    it = iter(iterable_2d)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Not all lists have same length!')
        
def replace_nan_amounts(amounts_df, value):
    """Will make all nan values in a dataframe become zero. Utilize this after determining the volume and mass of components, 
    some componants will have unknown density leading to a nan value. This will cause errors hence the value must be replaced, to ignore
   calculation simply make the replacement value to be 0."""
    amounts_df_zeroed = amounts_df.fillna(value)
    return amounts_df_zeroed

def isolate_common_column(df, common_string):
    """Returns dataframe with only the columns which contain the common string provided. 
    This is useful when calling for only a certain group of common information such as stocks or component masses"""
    cols = df.columns
    common_string_cols = [col for col in cols if common_string in col]
    final_df = df.copy()[common_string_cols]
    return final_df

def find_best_df_match(df, string):
    """Returns the dataframe with only the columns which contain the string provided. 
    It is identical to isolate_common_column(), so need to consolidate and phase one out."""
    match = df[[col for col in df if string in col]]
    return match

def stock_dict_from_plan(plan):
    stock_names = plan['Stock Names']
    stock_units = plan['Stock Concentration Units']
    stock_concentrations = plan['Stock Concentrations']
    stock_densities = plan['Stock Density (g/mL) (only for wtf)']
    stock_dict = stock_dictionary(stock_names, stock_units, stock_concentrations, stock_densities)

    return stock_dict

def find_component_column(component_name, df):
    df_columns = df.columns
    df_col_match = [col for col in df_columns if component_name == identify_component_name(col)]
    # add assertion if fail 
    return df_col_match

def calculate_total_stock_volumes(complete_df):
    stock_volumes = isolate_common_column(complete_df, 'stock')
    complete_df['Total Volume mL'] = stock_volumes.sum(axis=1)
    return complete_df

def convert_mL_to_uL(volumes_df): # switch this to be the dictionary using https://stackoverflow.com/questions/45468630/change-column-names-in-pandas-dataframe-from-a-list
    volumes_df = volumes_df.copy()
    columns_mL = [col for col in volumes_df if 'mL' in col]
    columns_uL = [col_mL.replace('mL', 'uL') for col_mL in columns_mL]
    volumes_uL = volumes_df[columns_mL]*1000

    volumes_df[columns_mL] = volumes_uL
    replace_dict = {mL:uL for uL, mL in zip(columns_uL, columns_mL)}
    volumes_df_uL = volumes_df.rename(columns=replace_dict)

    volumes_df_uL

    return volumes_df_uL

def remove_duplicates(df, sigfigs):
    df = df.round(sigfigs)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def filter_total_volume_restriction(df, max_total_volume):
    column_names = df.columns
    stock_column_names = [column_name for column_name in column_names if "stock" in column_name]
    stocks = df[stock_column_names]
    df['Total Volume'] = stocks.sum(axis=1)
    df = df[df['Total Volume']  <= max_total_volume]
    if df.empty is True:
        raise AssertionError("No suitable samples available to create due to TOTAL SAMPLE VOLUME being too high, reconsider labware or total sample mass/volume")
    return df
 
def filter_general_max_restriction(df, max_value, column_name):
    df_unfiltered = df.copy()
    df = df[df[column_name] <= max_value]
    if df.empty is True:
        raise AssertionError("No suitable samples available to create due to general filter being to low")
    return df

def filter_general_min_pipette_restriction(df, min_pipette_volume):
    column_names = df.columns
    stock_column_names = [column_name for column_name in column_names if "stock" in column_name]
    df_unfiltered = df.copy()
    
    for i, stock_column in enumerate(stock_column_names):
        df = df[df[stock_column] >= 0] # filtering all samples less than 0 
        if df.empty is True:
                raise AssertionError(stock_column + ' volumes contains only negative volumes. df series printed below', df_unfiltered[stock_column])

        df = df[(df[stock_column] >= min_pipette_volume) | (df[stock_column] == 0)] # filtering all samples that are less than miniumum pipette value and are NOT zero
        if df.empty is True:
            raise AssertionError(stock_column + ' volumes are below the pipette minimum of' + str(min_pipette_volume) + 'df series printed below', df_unfiltered[stock_column])

     
    return df 

##################### In progress ##############################

def concentration_from_linspace_all_info(plan, unity_filter = False, component_spacing = 'linear'): # if you go this route you can do whole dataframe operation you just need to verify all component units of the same type
    """ Uses linspaces to create a mesh of component concentrations
    """
    component_linspaces = plan['Component Concentration Linspaces [min, max, n]']

    conc_range_list = [] 
    for conc_linspace in component_linspaces:
        if component_spacing_type == "linear": 
            conc_range_list.append(np.linspace(*conc_linspace))
        elif component_spacing_type == "log": 
            conc_range_list.append(np.logspace(*conc_linspace))
    conc_grid = np.meshgrid(*conc_range_list)
    
    total_sample_amount = plan['Sample Amount']
    total_sample_amount_unit = plan['Sample Unit']
    component_names = plan['Component Shorthand Names']
    component_units = plan['Component Concentration Units']
    
    data = []
    columns = []
    for component_index in range(len(conc_grid)): 
        n = len(conc_grid[component_index].ravel())
        
        component_name_entry = [component_names[component_index]]*n
        columns.append('Component ' + str(component_index) + ' Name')
        data.append(component_name_entry)
        
        component_unit_entry = [component_units[component_index]]*n
        columns.append('Component ' + str(component_index) + ' Concentration Unit')
        data.append(component_unit_entry)
        
        component_concentration_column = 'Component ' + str(component_index) + ' Concentration Value'
        component_concetration_values = conc_grid[component_index].ravel()
        columns.append(component_concentration_column)
        data.append(component_concetration_values)
        
    component_conc_df = pd.DataFrame(data, columns).T # will terminate here if not needed unity
        
    if unity_filter == True: # generalize this  and make into callable function
        final_component_index = component_index + 1 
        
        component_name_entry = [component_names[final_component_index]]*n
        columns.append('Component ' + str(final_component_index) + ' Name')
        data.append(component_name_entry)
        
        component_unit_entry = [component_units[final_component_index]]*n
        columns.append('Component ' + str(final_component_index) + ' Concentration Unit')
        data.append(component_unit_entry)
        
        concentration_values_isolated = component_conc_df[[col for col in component_conc_df.columns if 'Concentration Value' in col]]
        completing_concentration_values = (1 - concentration_values_isolated.sum(axis=1)).tolist()
        data.append(completing_concentration_values)
        columns.append('Component ' + str(final_component_index) + ' Concentration Value')

    component_conc_df = pd.DataFrame(data, columns).T
    
    component_conc_df.insert(loc=0, column = 'Total Sample Amount Unit', value = [total_sample_amount_unit]*n)
    component_conc_df.insert(loc=0, column = 'Total Sample Amount', value = [total_sample_amount]*n) # this needs to be added at the same amount

    return component_conc_df

def determine_concentration_path(concentration_variable, variable_type):
    """ Determines the appropiate path to handle and create concentration design space... Still in progress requires kwargs
    """
    if 'variable_type' == 'csv':
        return cconcentration_variable
    elif 'variable_type' == 'excel':
        return concentration_variable
    elif variable_type == 'linspace':
        pass
    elif variable_type == 'sublists':
        pass

def determine_unit_pathway(plan, concentration_df):
    components_concentration_units = plan['Component Concentration Units']

    pass

def find_best_header_match(df, string):
    """Returns the dataframe column name which contains the provided string. Very unspecific so 
    be very careful with use."""
    
    match = [col for col in df if string in col][0]
    return match

def calculate_stock_prep_df(plan, volume_df, stock_dict, buffer_pct = 40):
    
    # Isolate all stock volume entries in dataframe
    cols = volume_df.columns
    stock_cols = [col for col in cols if "stock" in col]
    stock_df = volume_df.copy()[stock_cols]
    
    # Compound volumes and add buffer
    stock_df.loc['Total Volume'] = stock_df.sum(numeric_only=True, axis=0)*(1+(buffer_pct/100))
    prep_df = pd.DataFrame(stock_df.loc['Total Volume']).T
    
    # Ensure all unit are same then convet to liters for calculations, latter is not 100% necessary
    check_unit_congruence(prep_df)
    prep_df = convert_to_liter(prep_df)
    
    # Add the concentration and respective units (may have this be arguments instead since only would be +1)
    prep_df.loc['Final Selected Stock Concentrations'] = experiment_dict['Final Selected Stock Concentrations']
    prep_df.loc['Stock Concentration Units'] = experiment_dict['Stock Concentration Units']
    
    chem_database_df = pd.read_excel(chem_database_path)
    
    prep_dicts = {}
    for stock in prep_df:
        total_volume = prep_df[stock]['Total Volume']
        stock_unit = prep_df[stock]['Stock Concentration Units']
        stock_conc = prep_df[stock]['Final Selected Stock Concentrations']
        solutes, solvent = stock_components(stock) # currently only one solvent and solute supported

        #All stocks will obvi have a solvent, but the solute is optional
        solvent_component_info = chem_database_df.loc[chem_database_df['Component Abbreviation'] == solvent]
        solvent_density = solvent_component_info['Density (g/L)'].iloc[0]

        if not solutes: # if no solutes present
            solute_mass = 0
            solute_volume = 0
            solvent_volume = total_volume
            solvent_mass = solvent_volume*solvent_density
            prep_dict = {'solute mass g': solute_mass,
               'solute volume L': solute_volume,
               'solvent mass g': solvent_mass,
               'solvent volume L': solvent_volume}

        if solutes: 
            solute = solutes[0]
            solute_component_info = chem_database_df.loc[chem_database_df['Component Abbreviation'] == solute] # add assertion to ensure component in database

            if stock_unit == 'molarity':
                solute_mw = solute_component_info['Molecular Weight (g/mol)'].iloc[0] # only call info if needed, if common between units then pull up one level
                solute_density = solute_component_info['Density (g/L)'].iloc[0]
                prep_dict = stock_molarity(total_volume, stock_conc, solute_mw, solute_density, solvent_density)

            if stock_unit == 'wtf':
                # since no density data available at the moment need to rough estimate, does not matter since the mass is scaled according to wtf, so long as more.
                solute_density = solute_component_info['Density (g/L)'].iloc[0]
                density_mix = bimixture_density_wtf(stock_conc, solute_density, solvent_density)
                total_mass = total_volume*density_mix
                prep_dict = stock_wtf(total_mass, stock_conc, 1-stock_conc, solute_density, solvent_density)
        prep_dicts[stock] = prep_dict
    stock_prep_df = pd.DataFrame.from_dict(prep_dicts) # add total volumes
    stock_complete_df = pd.concat([prep_df,stock_prep_df])
    return stock_prep_df