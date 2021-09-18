import numpy as np
import matplotlib.pyplot as plt

from pyGDM2 import (core, propagators, fields, 
                    materials, linear, structures, 
                    tools, visu)

import os, shutil
wl = np.linspace(400, 1000, 5)


def uvvis(wl, shape, geom, medium):
        step = 5
        geom_list = []
        
        if shape=='sphere':
            geometry = structures.sphere(step, **geom)
            
        elif shape=='cylinder':
            geometry = structures.nanorod(step, **geom)
            
        else:
            raise RuntimeError('Shape %s is not recognized'%shape)
            
        struct = structures.struct(step, geometry, 
            materials.gold(), verbose=False)
        
        field_generator = fields.plane_wave
        kwargs = dict(theta=0, inc_angle=180)

        efield = fields.efield(field_generator,
                       wavelengths=wl, kwargs=kwargs)

        dyads = propagators.DyadsQuasistatic123(**medium)

        sim = core.simulation(struct, efield, dyads)
        sim.scatter(verbose=False)
        field_kwargs = tools.get_possible_field_params_spectra(sim)

        config_idx = 0
        wl, spectrum = tools.calculate_spectrum(sim,
                            field_kwargs[config_idx], linear.extinct)
        
        abs_ = spectrum.T[2]/np.max(spectrum.T[2])
        
        return abs_ 

savedir = './model_data/'
savedir = os.path.abspath(savedir)
if  os.path.exists(savedir):
    shutil.rmtree(savedir)
os.makedirs(savedir)
print(savedir)

def run(shape, geom, medium, params, count):
    try:
        abs_ = uvvis(wl , shape, geom=geom, medium=medium)
        print(' is a success', end='\n')
        np.savez(savedir+'%d'%(count)+'.npz', abs_, params)
    except Exception as e:
        raise e
        print(' is a disater', end='\n')


count = 0
print('count\t radius n1 n2 shape length')
for radius in 0.2*np.linspace(10,100, num=20):
    for n1 in np.linspace(1,2,num=10):
        for n2 in np.linspace(1,2, num=10):
            medium={'n1':n1, 'n2':n2}
            for shape in ['sphere', 'cylinder']:    
                if shape=='sphere':
                    geom = {'R':radius, 'mesh':'hex'}
                    params = [radius, n1, n2, shape, 0]
                    print('%d\t%s'%(count, params), end='')
                    run(shape, geom , medium, params, count)
                    count += 1
                else:
                    for length in 0.2*np.linspace(10,200, num=20):
                        params = [radius, n1, n2, shape, length]
                        geom = {'R':radius,'L':length, 'mesh':'hex'}
                        print('%d\t%s'%(count, params), end='')
                        run(shape, geom , medium, params, count)
                        count += 1
    
    
    
    