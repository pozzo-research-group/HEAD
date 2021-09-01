import numpy as np
import head 
from configparser import ConfigParser
import os, sys, pdb
import torch
import logging
logging.basicConfig(level=logging.INFO, 
	format='%(asctime)s%(message)s ')

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	
savedir = config['Default']['savedir']
iteration = config['BO']['iteration']
logging.info('\tGenerating spectra from iteration %d'%iteration)
spectra_dir = savedir+'spectra_%d'%iteration


if os.path.exists(spectra_dir):
	raise RuntimeError('It appears that spectra for %d iteration has already been collected'%iteration)
else:
    os.makedirs(spectra_dir)

if __name__=='__main__':
	new_x = torch.load(savedir+'candidates_%d.pt'%iteration, 
		map_location=tkwargs['device'])
	logging.info('\tCandidates %s'%(new_x))
	for i, x in enumerate(new_x.squeeze(1)):
	    x_np = x.numpy()
	    sim = head.Emulator(n_structures=config['Modelling']['n_structures'])
	    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
	    
	    if 'saxs' in config['BO']['objective']:
	    	q, si = sim.get_saxs(config['Modelling']['n_sas_samples'])
	    	np.savetxt(spectra_dir+'/%d_saxs.txt'%i, si, delimiter=',')
	    	logging.info('\tSAXS for point %d: %s; q: [%f,%f] ; si: [%f, %f] '%(i, x_np, min(q), max(q), min(si), max(si)))
	    if 'uvvis' in config['BO']['objective']:
	    	wl, Ii = sim.get_spectrum(config['Modelling']['n_uvvis_samples'])
	    	logging.info('\tUV-Vis for point %d: %s; wl: [%f,%f] ; Ii: [%f, %f] '%(i, x_np, min(wl), max(wl), min(Ii), max(Ii)))
	    	np.savetxt(spectra_dir+'/%d_uvvis.txt'%i, Ii, delimiter=',')

	logging.info('\tspectra collected using a simulator for iteration %d'%iteration)
	
