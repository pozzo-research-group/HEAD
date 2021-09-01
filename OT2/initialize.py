import head
import numpy as np
import os, pdb
import logging
import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
savedir = config['Default']['savedir']	
logging.basicConfig(level=logging.INFO, 
	format='%(asctime)s%(message)s \t')

def make_grid():
	X = np.linspace(10,30, num=int(config['Default']['num_grid_perdim'])) 
	Y = np.linspace(1e-3,1, num=int(config['Default']['num_grid_perdim']))
	grid = head.Grid(X,Y)
	logging.info('\tNumber of grid points' + str(grid.shape))
	np.savetxt(savedir+'grid.txt', grid, delimiter=',')
	
def make_target():
	sim = head.Emulator(n_structures=int(config['Modelling']['n_structures']))
	sim.make_structure(r_mu=float(config['Default']['r_mu']),
		r_sigma=float(config['Default']['r_sigma']))
	
	logging.info('\tTargets are : '+','.join(i for i in config['BO']['objective']))
	if 'saxs' in config['BO']['objective']:
		q, st = sim.get_saxs(n_samples=config['Modelling']['n_sas_samples'])
		np.savetxt(savedir+'target_saxs.txt', st, delimiter=',')
		logging.info('\tSAXS Target q: [%f,%f] ; st: [%f, %f] '%(min(q), max(q), min(st), max(st)))
		np.savetxt(savedir+'q.txt', q, delimiter=',')

	if 'uvvis' in config['BO']['objective']:
		wl, It = sim.get_spectrum(n_samples=config['Modelling']['n_uvvis_samples'])
		logging.info('\tUVVIS Target wl: [%f,%f] ; Ii: [%f, %f] '%(min(wl), max(wl), min(It), max(It)))
		np.savetxt(savedir+'/target_uvvis.txt', It, delimiter=',')
		np.savetxt(savedir+'wl.txt', wl, delimiter=',')

if __name__=='__main__':
	
	if  os.path.exists(savedir):
		logging.error('\tRequired directory already exists...\n'
			'Confirm that this is not a duplicate run, manually delete the directory and re-run again')
	else:
		os.makedirs(savedir)
		logging.info('\tMade the directory for this experiment %s'%savedir)
		make_grid()
		
		logging.info('\tMaking the target responses using the Emulator')
		make_target()
		config['BO']['iteration'] = 0
		with open(os.path.abspath('./config.yaml'), 'w') as fp:
			yaml.dump(config, fp)
			
		logging.info('\tInitialization is successful, current iteration %d'%config['BO']['iteration'])
		
		
		
		