import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyGDM2 import  (structures, materials, core, 
                    linear, fields, propagators, 
                    tools)

def get_spectrum(geometry, step, wavelengths):
    '''Obtains a uv-vis spectra for a specified geometry'''
    material = materials.gold()
    struct = structures.struct(step, geometry, material, verbose=False)
    struct = structures.center_struct(struct)
    field_generator = fields.plane_wave
    kwargs = dict(theta=0, inc_angle=180)

    efield = fields.efield(field_generator,
                   wavelengths=wavelengths, kwargs=kwargs)
    
    dyads = propagators.DyadsQuasistatic123(n1 = 1.33, n2 = 1.33, n3 = 1.33)

    sim = core.simulation(struct, efield, dyads)
    sim.scatter(verbose=False)
    field_kwargs = tools.get_possible_field_params_spectra(sim)

    config_idx = 0
    wl, spectrum = tools.calculate_spectrum(sim,
                        field_kwargs[config_idx], linear.extinct)
    
    abs_ = spectrum.T[2]/np.max(spectrum.T[2])
    return abs_, geometry

def obtain_spectra(step, radius_mean, radius_std, wavelength):
    '''Calculates the absorption spectra of polydisperse gold spheres that have a normally distributed
       radius.
       Inputs:
       - step: The step size used for the calculation.
       - radius_mean: The mean of the normal distribution used to calculate the radius of the sphere
       - radius_std: The std of the normal distribution used to calculate the radius of the sphere
       - wavelength: A 1-d array of the wavelength values to calculate the absorption spectra
       Outputs:
       - array: A 2d array of the wavelengths and Intensity values.  
    '''
    n_spheres = 7
    radius_list = []
    for i in range(n_spheres):
        # Normal distribution parameters for Sphere Radius 
        radius_mean = 6
        radius_std = 3
        r = (np.random.randn(1)[0]*radius_std + radius_mean)/step
        radius_list.append(r)
        geometry = structures.sphere(step, R=r, mesh='cube')
        loc_array = np.array([[0,0,0],[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
        sphere = np.hstack((geometry[:,0].reshape(-1,1) + 30*loc_array[i,0]*radius_mean, geometry[:,1].reshape(-1,1) + 30*loc_array[i,1]*radius_mean, geometry[:,2].reshape(-1,1)+ 30*loc_array[i,2]*radius_mean))
        if i == 0:
            sample = sphere
        else:
            sample = np.vstack((sample, sphere))
    I, g = get_spectrum(geometry, step, wavelength)
    array = np.hstack((wavelength.reshape(-1,1), I.reshape(-1,1)))
    return array
