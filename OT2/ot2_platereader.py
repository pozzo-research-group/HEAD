import numpy as np
import head 
import os, sys, pdb
import torch
import logging
import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import logger
logger = logger('ot2_platereader')

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

with open(os.path.abspath('./config.yaml'), 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	
savedir = config['Default']['savedir']
iteration = config['BO']['iteration']
logger.info('Generating spectra from iteration %d'%iteration)

if __name__=='__main__':
	new_x = torch.load(savedir+'candidates_%d.pt'%iteration, 
		map_location=tkwargs['device'])
	logger.info('Candidates %s'%(new_x))
	saxs, uvvis = [], []
	for i, x in enumerate(new_x.squeeze(1)):
	    x_np = x.numpy()
	    sim = head.Emulator(n_structures=config['Modelling']['n_structures'])
	    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
	    if 'saxs' in config['BO']['objective']:
	    	q, si = sim.get_saxs(config['Modelling']['n_sas_samples'])
	    	saxs.append(si)
	    	logger.info('SAXS for point %d: %s; q: [%f,%f] ; si: [%f, %f] '%(i, x_np, min(q), max(q), min(si), max(si)))
	    if 'uvvis' in config['BO']['objective']:
	    	wl, Ii = sim.get_spectrum(config['Modelling']['n_uvvis_samples'])
	    	uvvis.append(Ii)
	    	logger.info('UV-Vis for point %d: %s; wl: [%f,%f] ; Ii: [%f, %f] '%(i, x_np, min(wl), max(wl), min(Ii), max(Ii)))
	
	if len(saxs)>0 : pd.DataFrame(saxs).T.to_excel(savedir+'/saxs_%d.xlsx'%iteration) 
	if len(uvvis)>0 : pd.DataFrame(uvvis).T.to_excel(savedir+'/uvvis_%d.xlsx'%iteration)
	
	logger.info('spectra collected using a simulator for iteration %d'%iteration)
	
