import head
import numpy as np
import os, pdb, sys
import logging
import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
savedir = config['Default']['savedir']	


def make_grid():
	ctab = np.linspace(6.87e-5,0.109, num=int(config['Default']['num_grid_perdim'])) 
	agno3 = np.linspace(4e-5,0.0039, num=int(config['Default']['num_grid_perdim']))
	haucl4 = np.linspace(8e-7,0.0005, num=int(config['Default']['num_grid_perdim']))
	acid = np.linspace(0.0004,0.04, num=int(config['Default']['num_grid_perdim']))
	seeds = np.linspace(5.84e-9,5.84e-7, num=int(config['Default']['num_grid_perdim']))
	
	grid = head.Grid(ctab, agno3, haucl4, acid, seeds)
	logger.info('Number of grid points' + str(grid.shape))
	np.savetxt(savedir+'grid.txt', grid, delimiter=',')
	
def make_target():
	sim = head.Emulator(n_structures=int(config['Modelling']['n_structures']))
	sim.make_structure(r_mu=float(config['Default']['r_mu']),
		r_sigma=float(config['Default']['r_sigma']))
	
	logger.info('Targets are : '+','.join(i for i in config['BO']['objective']))
	if 'saxs' in config['BO']['objective']:
		q, st = sim.get_saxs(n_samples=config['Modelling']['n_sas_samples'])
		np.savetxt(savedir+'target_saxs.txt', st, delimiter=',')
		logger.info('SAXS Target q: [%f,%f] ; st: [%f, %f] '%(min(q), max(q), min(st), max(st)))
		np.savetxt(savedir+'q.txt', q, delimiter=',')

	if 'uvvis' in config['BO']['objective']:
		wl, It = sim.get_spectrum(n_samples=config['Modelling']['n_uvvis_samples'])
		logger.info('UVVIS Target wl: [%f,%f] ; Ii: [%f, %f] '%(min(wl), max(wl), min(It), max(It)))
		np.savetxt(savedir+'/target_uvvis.txt', It, delimiter=',')
		np.savetxt(savedir+'wl.txt', wl, delimiter=',')

def make_target_from_file(file,name):
	spectra = pd.read_excel(args.xlsx, index_col=0, engine='openpyxl') 
	codomain = spectra.iloc[:,1]
	domain = spectra.iloc[:,0]
	np.savetxt(savedir+'target_%s.txt'%name, codomain, delimiter=',')
	logger.info('%s Target domain: [%f,%f] ; codomain: [%f, %f] '%(name, min(q), max(q), min(st), max(st)))
	np.savetxt(savedir+'%s_domain.txt'%name, domain, delimiter=',')

if __name__=='__main__':
	
	if  os.path.exists(savedir):
		raise RuntimeError('Required directory already exists...\n'
			'Confirm that this is not a duplicate run, manually delete the directory and re-run again')
	else:
		os.makedirs(savedir)
		sys.path.append(os.path.join(os.path.dirname('./utils.py')))
		from utils import logger
		logger = logger('initialize')
		logger.info('Made the directory for this experiment %s'%savedir)
		make_grid()
		
		logger.info('Making the target responses using the Emulator')
		make_target()
		config['BO']['iteration'] = 0
		with open(os.path.abspath('./config.yaml'), 'w') as fp:
			yaml.dump(config, fp)
			
		logger.info('Initialization is successful, current iteration %d'%config['BO']['iteration'])
		
		
		
		