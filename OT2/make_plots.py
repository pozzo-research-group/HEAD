import os, sys, pdb
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import torch
import head
import json

plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 18
                    }
                   )

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
savedir = config['Default']['savedir']
figdir = savedir+'figures/'

if not os.path.exists(figdir):
	os.makedirs(figdir)

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import logger
logger = logger('make_plots')

# load all required variables
train_obj = torch.load(savedir+'train_obj.pt')
train_x = torch.load(savedir+'train_x.pt')

sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
from run_bo import load_models

mll, model = load_models(train_x, train_obj)

batch_number = torch.cat(
    [torch.zeros(config['BO']['n_init_samples']), 
    torch.arange(1, config['BO']['iteration']+1).repeat(config['BO']['batch_size'], 1).t().reshape(-1)]).numpy()

# 1. Plot paretofront
def paretofront():
	fig, axes = plt.subplots(1, 1)
	cm = plt.cm.get_cmap('viridis')

	sc = axes.scatter(train_obj[:, 0].cpu().numpy(), train_obj[:,1].cpu().numpy(), 
	    c=batch_number, alpha=0.8,
	)
	axes.set_xlabel("SAS similarity")
	axes.set_ylabel("UVVis similarity")
	norm = plt.Normalize(batch_number.min(), batch_number.max())
	sm =  ScalarMappable(norm=norm, cmap=cm)
	sm.set_array([])
	cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
	cbar = fig.colorbar(sm, cax=cbar_ax)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('iteration', rotation=270)
	plt.savefig(figdir + '/paretofront.png', bbox_inches='tight')
	plt.close()


# 2. Plot batchwise trace
def trace(objs):
	spectra_files = glob.glob(savedir + 'spectra_*')
	if 'saxs' in objs:
		q = np.loadtxt(savedir+'q.txt', delimiter=',')
		
	if 'uvvis' in objs:
		wl = np.loadtxt(savedir+'wl.txt', delimiter=',')
		
	for b,spectra_dir in enumerate(spectra_files):
		fig, ax = plt.subplots()
		Yb = train_obj[batch_number==b].cpu().numpy()
		files = glob.glob(spectra_dir + '/*.txt')
		if len(objs)==1:
			for i, file in enumerate(files):
				if 'saxs' in objs:
					si = np.loadtxt(file, delimiter=',')
					ax.loglog(q,si,label='%.2f'%(Yb[i,0]))
				elif 'uvvis' in objs:
					Ii = np.loadtxt(file, delimiter=',')
					ax.plot(wl, Ii, label='%.2f'%(Yb[i,0]))
				elif 'dls' in objs:
					raise NotImplementedError
			ax.legend()	
		else:
			n_files = len(files)//2		
			fig, axs = plt.subplots(1,len(objs),figsize=(5*len(objs), 5))
			fig.suptitle('Batch number %d'%b)
			fig.subplots_adjust(wspace=0.2)

			for i in range(n_files):
				if 'saxs' in objs:
					si = np.loadtxt(spectra_dir+'/%d_saxs.txt'%i, delimiter=',')
					try:
						Yb[i,0]
					except:
						pdb.set_trace()
					axs[0].loglog(q, si, label='%.2f'%(Yb[i,0]))
				if 'uvvis' in objs:
					Ii = np.loadtxt(spectra_dir+'/%d_uvvis.txt'%i, delimiter=',')
					axs[1].plot(wl, Ii, label='%.2f'%(Yb[i,1]))
				if 'dls' in objs:
					raise NotImplementedError
		    	
			for ax in axs:
				ax.legend()	
		fname = figdir + 'trace_b%d'%b
		logger.info('\tPlotted batch number %d in %s'%(b, fname))    
		plt.savefig(fname, bbox_inches='tight')
		plt.close()


# 3. Plot the optimal curves
def optimal():
	fig, axs = plt.subplots(1,3,figsize=(4*3,4))


	sim = head.Emulator(n_structures=config['Modelling']['n_structures'])

	sim.make_structure(r_mu=config['Default']['r_mu'],
		r_sigma=float(config['Default']['r_sigma']))
	line_target = sim.plot_radii(axs[0])

	sim.make_structure(r_mu=opt_x[0],r_sigma=opt_x[1])
	line_best = sim.plot_radii(axs[0])
	axs[0].legend([line_target, line_best],['Target', 'Best'])
	axs[0].set_xlabel('radius')
	axs[0].set_ylabel('Probability density')

	q, sopt = sim.get_saxs(n_samples=config['Modelling']['n_sas_samples'])
	st = np.loadtxt(savedir+'target_saxs.txt', delimiter=',')
	axs[1].loglog(q, st, label='Target')
	axs[1].loglog(q, sopt, label='Optimal')
	axs[1].legend()

	wl, Iopt = sim.get_spectrum(n_samples=config['Modelling']['n_uvvis_samples'])
	It = np.loadtxt(savedir+'target_uvvis.txt', delimiter=',')
	axs[2].plot(wl,It, label='Target')
	axs[2].plot(wl,Iopt, label='Optimal')
	axs[2].legend()

	fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))
	plt.savefig(figdir + '/optimums.png', bbox_inches='tight')
	plt.close()

if __name__=='__main__':
	objs = config['BO']['objective']
	trace(objs)
	if len(objs)>1:
		paretofront()
	
	





