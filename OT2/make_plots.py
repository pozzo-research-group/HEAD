import os, sys, pdb
import numpy as np
import glob
from configparser import ConfigParser
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import torch
import head

plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 18
                    }
                   )

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
figdir = savedir+'figures/'

if not os.path.exists(figdir):
	os.makedirs(figdir)

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

# load all required variables
train_obj = torch.load(savedir+'train_obj.pt')
train_x = torch.load(savedir+'train_x.pt')

sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
from run_bo import load_models

mll, model = load_models(train_x, train_obj)


# 1. Plot paretofront
fig, axes = plt.subplots(1, 1)
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(int(config['BO']['n_init_samples'])), 
    torch.arange(1, int(config['BO']['iteration'])+1).repeat(int(config['BO']['batch_size']), 1).t().reshape(-1)]).numpy()


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
spectra_files = glob.glob(savedir + 'spectra_*')
q = np.loadtxt(savedir+'q.txt', delimiter=',')
wl = np.loadtxt(savedir+'wl.txt', delimiter=',')
for b,spectra_dir in enumerate(spectra_files):
	obj = train_obj[batch_number==b].cpu().numpy()
	fig, axs = plt.subplots(1,2,figsize=(5*2, 5))
	fig.suptitle('Batch number %d'%b)
	fig.subplots_adjust(wspace=0.2)
	files = glob.glob(spectra_dir + '/*.txt')
	n_files = len(files)//2
	for i in range(n_files):
		si = np.loadtxt(spectra_dir+'/%d_saxs.txt'%i, delimiter=',')
		Ii = np.loadtxt(spectra_dir+'/%d_uvvis.txt'%i, delimiter=',')
		axs[0].loglog(q, si, label='%.2f'%(obj[i,0]))
		axs[1].plot(wl, Ii, label='%.2f'%(obj[i,1]))
    	
	for ax in axs:
		ax.legend()	
	    
	plt.savefig(figdir + '/trace_b%d.png'%b, bbox_inches='tight')
	plt.close()


# 3. Plot the optimal curves
from botorch.acquisition import PosteriorMean
from botorch.acquisition.objective import ScalarizedObjective
sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
from run_bo import selector
objective = ScalarizedObjective(weights=torch.tensor([0.5, 0.5]).to(**tkwargs))

opt_x = selector(PosteriorMean(model, objective=objective), q=1)
opt_x = opt_x.cpu().numpy().squeeze()

print('Best sample from the optimization :', opt_x)

fig, axs = plt.subplots(1,3,figsize=(4*3,4))


sim = head.Emulator(n_structures=int(config['Modelling']['n_structures']))

sim.make_structure(r_mu=float(config['Default']['r_mu']),
	r_sigma=float(config['Default']['r_sigma']))
line_target = sim.plot_radii(axs[0])

sim.make_structure(r_mu=opt_x[0],r_sigma=opt_x[1])
line_best = sim.plot_radii(axs[0])
axs[0].legend([line_target, line_best],['Target', 'Best'])
axs[0].set_xlabel('radius')
axs[0].set_ylabel('Probability density')

q, sopt = sim.get_saxs(n_samples=int(config['Modelling']['n_sas_samples']))
st = np.loadtxt(savedir+'target_saxs.txt', delimiter=',')
axs[1].loglog(q, st, label='Target')
axs[1].loglog(q, sopt, label='Optimal')
axs[1].legend()

wl, Iopt = sim.get_spectrum(n_samples=int(config['Modelling']['n_uvvis_samples']))
It = np.loadtxt(savedir+'target_uvvis.txt', delimiter=',')
axs[2].plot(wl,It, label='Target')
axs[2].plot(wl,Iopt, label='Optimal')
axs[2].legend()

fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))
plt.savefig(figdir + '/optimums.png', bbox_inches='tight')
plt.close()







