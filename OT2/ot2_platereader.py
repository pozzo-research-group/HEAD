import numpy as np
import head 
from configparser import ConfigParser
import os, sys
import torch

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import ground_truth

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
iteration = int(config['BO']['iteration'])
print('Generating spectra from %d'%iteration)
spectra_dir = savedir+'spectra_%d'%iteration

if os.path.exists(spectra_dir):
	raise RuntimeError('It appears that spectra for %d iteration has already been collected'%iteration)
else:
    os.makedirs(spectra_dir)

if __name__=='__main__':
	new_x = torch.load(savedir+'candidates_%d.pt'%iteration, 
		map_location=tkwargs['device'])
	
	for i, x in enumerate(new_x.squeeze(1)):
	    x_np = x.numpy()
	    sim = head.Emulator(n_structures=int(config['Modelling']['n_structures']))
	    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
	    
	    q, si = sim.get_saxs(int(config['Modelling']['n_sas_samples']))
	    wl, Ii = sim.get_spectrum(int(config['Modelling']['n_uvvis_samples']))
	    
	    np.savetxt(spectra_dir+'/%d_saxs.txt'%i, si, delimiter=',')
	    np.savetxt(spectra_dir+'/%d_uvvis.txt'%i, Ii, delimiter=',')

	print('spectra collected using a simulator for iteration', iteration)
	
	if iteration==0:
		train_obj = ground_truth(spectra_dir)
		torch.save(train_obj, savedir+'train_obj.pt')