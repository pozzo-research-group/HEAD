import numpy as np

def model(q, r_mean, r_std):
    '''Returns a model SAXS curve of spheres with a normally distributed mean of
       "r_mean" and standard deviation of "r_std".
       Inputs:
       q: q-range of the saxs curve, example: np.linsapce(1e-1, 1,1000)
       r_mean: sphere radisu mean, example: 6 nm
       r_std: sphere radius standard deviation, example: 1 nm'''
    R_list = []
    for i in range(100):
        R = np.random.randn(1)*r_std + r_mean
        R_list.append(R)
        F = 3*(np.sin(q*R) - q*R*np.cos(q*R))/(q*R)**3
        I = (F**2)
        I = I.reshape(-1,1)
        if i == 0:
            I_array = I
        else:
            I_array = np.hstack((I_array, I))
    return np.mean(I_array, axis=1)