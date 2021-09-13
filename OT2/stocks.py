import pandas as pd
import numpy as np

class conc_to_vol:
    '''Inputs in SI units, outputs in uL'''
    def __init__(self, **kwargs):
        self.conc = kwargs['conc_array']
        self.small_pipette = np.array(kwargs['small_pipette'])
        self.large_pipette = np.array(kwargs['large_pipette'])
        self.S1_conc = np.array(kwargs['S1_conc'])
        self.S2_conc = np.array(kwargs['S2_conc'])
        self.S3_conc = np.array(kwargs['S3_conc'])
        self.S4_conc = np.array(kwargs['S4_conc'])
        self.S5_conc = np.array(kwargs['S5_conc'])
        return 
    
    def single_stock_solution_conc(self, stock_conc1, stock_conc2, stock_conc3, stock_conc4, stock_conc5):
        '''stock_conc is a vector of the concentrations of one type of stock solution'''
        self.stock_conc1 = stock_conc1
        self.stock_conc2 = stock_conc2
        self.stock_conc3 = stock_conc3
        self.stock_conc4 = stock_conc4
        self.stock_conc5 = stock_conc5
    
    def pipette_range(self):
        '''small_pipette and large_pipette are vectors of the ranges of the two pipettes'''
        self.sp_UL = self.small_pipette[-1]
        self.sp_LL = self.small_pipette[0]
        self.lp_UL = self.large_pipette[-1]
        self.lp_LL = self.large_pipette[0]
    
    def conc_to_moles(self, conc):
        moles_array = conc*0.001750 #conc*1mL 
        self.m1 = moles_array[:,0]
        self.m2 = moles_array[:,1]
        self.m3 = moles_array[:,2]
        self.m4 = moles_array[:,3]
        self.m5 = moles_array[:,4]
    
    def initialize_volume_arrays(self, conc):
        self.s1_vol_array = np.zeros((conc.shape[0], len(self.stock_conc1)))
        self.s2_vol_array = np.zeros((conc.shape[0], len(self.stock_conc2)))
        self.s3_vol_array = np.zeros((conc.shape[0], len(self.stock_conc3)))
        self.s4_vol_array = np.zeros((conc.shape[0], len(self.stock_conc4)))
        self.s5_vol_array = np.zeros((conc.shape[0], len(self.stock_conc5)))    
            
    def calculate_volume(self, stock_concs, vol_array, moles_column):
        '''conc_column - stock solution concentration 1-D array 
            vol_array - initialized zeros array of volumes 
            moles_column - amount of moles in sample specified by BO algorithm'''
        for i in range(vol_array.shape[0]):
            next_row = False
            for j in range(vol_array.shape[1]):
                if next_row == True:
                    break
                for k in range(stock_concs.shape[0]):
                    volume = moles_column[i]/stock_concs[k]*10**6
                    if volume < self.lp_UL and volume > self.sp_LL:
                        next_row = True
                        vol_array[i,j] = volume
                    else:
                        j = j + 1
                        if j > vol_array.shape[1]-1:
                            raise Exception('Change the concentration of the stock solution or the pipette range.')
                        pass
                    if next_row == True:
                        break
        return vol_array
    
    def calculate_volumes(self):
        vol_c1 = self.calculate_volume(self.stock_conc1, self.s1_vol_array, self.m1)
        vol_c2 = self.calculate_volume(self.stock_conc2, self.s2_vol_array, self.m2)
        vol_c3 = self.calculate_volume(self.stock_conc3, self.s3_vol_array, self.m3)
        vol_c4 = self.calculate_volume(self.stock_conc4, self.s4_vol_array, self.m4)
        vol_c5 = self.calculate_volume(self.stock_conc5, self.s5_vol_array, self.m5)
        vol_array = np.hstack((vol_c1, vol_c2, vol_c3, vol_c4, vol_c5))
        return vol_array
    
    def vol_to_vol_frac(self, vol_array):
        for i in range(vol_array.shape[0]):
            row_sum = np.sum(vol_array[i,:])
            for j in range(vol_array.shape[1]):
                vol_array[i,j] = vol_array[i,j]/row_sum
        return vol_array
    
    def perform_calculation(self):
        self.conc_to_moles(self.conc)
        self.pipette_range()
        self.single_stock_solution_conc(self.S1_conc, self.S2_conc, self.S3_conc, self.S4_conc, self.S5_conc) #Units of M
        self.initialize_volume_arrays(self.conc)
        vol = self.calculate_volumes()
        #vol_frac = self.vol_to_vol_frac(vol)
        water = 1750 - np.sum(vol, axis = 1)
        water2 = []
        for i in range(water.shape[0]):
            if water[i] > 1000:
                water2.append(water[i]/2)
                water[i] = water[i]/2
            elif water[i] < 20:
                water[i] = 0
                water2.append(0)
            else:
                water2.append(0)
        water2 = np.asarray(water2)
        
        seeds = vol[:,-2:]
        vol = vol[:,0:-2]
        volumes = np.hstack((vol, water.reshape(-1,1), water2.reshape(-1,1)))
        column_titles = ['CTAB_A-stock','CTAB_B-stock','AG_A-stock','AG_B-stock','HAuCl_A-stock','HAuCl_B-stock','AA_A-stock','AA_B-stock','water1-stock','water2-stock']
        df = pd.DataFrame(volumes, columns = column_titles)
        df_seeds = pd.DataFrame(seeds, columns = ['Seeds_A-stock', 'Seeds_B-stock'])
        return df, df_seeds

# Run this code with different numbers to perform calculation
# Step 1: Enter concentration array. The number of columns is the number of stock solutions and the rows is the number of samples.
#conc = np.array([[16.8e-3, 0.084e-3, 0.09e-3, 0.0075e-3, 2.5e-8],[10.8e-3,0.184e-3,0.69e-3,0.075e-3,6.5e-8]]) #Units of M
# Step 2: Enter the pipette ranges and the concentrations of the stock solutions
#exp1 = conc_to_vol(conc_array = conc, small_pipette = [0,50], large_pipette = [50,300], S1_conc = [0.2,0.4], S2_conc = [0.001, 0.005], S3_conc = [0.078], S4_conc = [0.004], S5_conc = [0.000025])
# Step 3: Perform the concentration to volume calculation
#volumes = exp1.perform_calculation()


def to_volume(conc):
    exp1 = conc_to_vol(conc_array = conc, 
    small_pipette = [5,50], 
    large_pipette = [50,800], 
    S1_conc = [0.00601, 0.2406], 
    S2_conc = [0.0035, 0.035], 
    S3_conc = [7e-5, 1.74e-3], 
    S4_conc = [0.035, 0.35], 
    S5_conc = [2.04e-6, 2.04e-5])

    volume_df, seed_df = exp1.perform_calculation()
    
    return volume_df, seed_df

if __name__=='__main__':
    conc = np.array([[0.109, 0.0039, 5e-4, 0.04, 5.84e-7],
    [6.87e-5, 4e-5, 8e-7, 0.0004, 5.84e-9],
    [0.047, 0.0008, 0.000778, 0.0397, 4.15e-7]]) #Units of M
    volume_df, seed_df = to_volume(conc)
    print('Volumes :...\n', volume_df)
    print('Seeds: ...\n', seed_df)







