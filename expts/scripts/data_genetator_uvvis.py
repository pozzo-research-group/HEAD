import numpy as np
import matplotlib.pyplot as plt

from pyGDM2 import (propagators, 
                    materials, linear, structures, 
                    tools, visu)
                    
from pyGDM2 import fields_py as fields
from pyGDM2 import core_py as core

import os, shutil, time

NUM_WAVELENGTH_SAMPLES = 100
STEP = 10
NUM_VAR_SAMPLES = 5

savedir = './model_data/'
savedir = os.path.abspath(savedir)
if  os.path.exists(savedir):
    shutil.rmtree(savedir)
os.makedirs(savedir)
print('Data will be saved into : ', savedir)

wl = np.linspace(400, 1000, NUM_WAVELENGTH_SAMPLES)
np.savez(savedir+'/wavelength.npz', wl)

def uvvis(wl, shape, geom, medium):
        if shape=='sphere':
            geometry = structures.sphere(STEP, **geom)
            
        elif shape=='cylinder':
            geometry = structures.nanorod(STEP, **geom)
            
        else:
            raise RuntimeError('Shape %s is not recognized'%shape)
        struct = structures.struct(STEP, geometry, 
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
        _, spectrum = tools.calculate_spectrum(sim,
                            field_kwargs[config_idx], linear.extinct)
        
        abs_ = spectrum.T[2]/np.max(spectrum.T[2])
                
        return abs_ 




def run(shape, geom, medium, params, count):
	try:
		t0 = time.time()
		abs_ = uvvis(wl , shape, geom=geom, medium=medium)
		if np.isnan(abs_).any():
			raise RuntimeError
		tf = time.time()
		print(' is a success and took %.2f s'%(tf-t0), end='\n')
		np.savez(savedir+'/%d'%(count)+'.npz', abs_, params)
	except Exception as e:
		raise e
		print(' is a disater', end='\n')
		
count = 0
print('count\t radius n1 n2 shape length')

def compute_total():
	count = 0
	for radius in np.linspace(0.05,2.5, num=NUM_VAR_SAMPLES):
		for n1 in np.linspace(1,2,num=NUM_VAR_SAMPLES):
			for n2 in np.linspace(1,2, num=NUM_VAR_SAMPLES):
				medium={'n1':n1, 'n2':n2}
				for shape in ['sphere', 'cylinder']:    
					if shape=='sphere':
						count += 1
					else:
						for length in np.linspace(0.05,2.5, num=NUM_VAR_SAMPLES):
							count += 1
	return count

print('Total of %d spectra are requested'%compute_total())

for radius in np.linspace(10,100, num=NUM_VAR_SAMPLES):
    for n1 in np.linspace(1,2,num=NUM_VAR_SAMPLES):
        for n2 in np.linspace(1,2, num=NUM_VAR_SAMPLES):
            medium={'n1':n1, 'n2':n2}
            for shape in ['sphere', 'cylinder']:    
                if shape=='sphere':
                    geom = {'R':radius/STEP, 'mesh':'hex'}
                    params = [radius, n1, n2, shape, 0]
                    print('%d\t%s'%(count, params), end='')
                    run(shape, geom , medium, params, count)
                    count += 1
                else:
                    for length in np.linspace(50,400, num=NUM_VAR_SAMPLES):
                        params = [radius, n1, n2, shape, length]
                        geom = {'R':radius/STEP,'L':length/STEP, 'mesh':'hex'}
                        print('%d\t%s'%(count, params), end='')
                        run(shape, geom , medium, params, count)
                        count += 1
    
print('Generated a total of %d spectra'%count)
    
    