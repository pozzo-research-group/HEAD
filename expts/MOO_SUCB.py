#!/usr/bin/env python
# coding: utf-8

# ## Bayesian Optimization with multiple objectives
# Our pipeline should be as follows:
# 1. Define our design space as a grid or hyperplane etc
# 2. Define a model as surrogate to compute a score between target and a response query
# 3. Define acquistion function to score candidates
# 4. Define a selector to select candidate points
# 4. Define the optimization routine for the problem

# In[1]:


import os, shutil
import torch
import time
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
import pdb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
import head
import numpy as np
from head.metrics import euclidean_dist
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import LinearMCObjective
from matplotlib.cm import ScalarMappable

N_UVVIS_SAMPLES = 50
N_SAS_SAMPLES = 50
NUM_GRID_PERDIM = 20
BATCH_SIZE = 5
N_ITERATIONS = 2
R_mu = 20
R_sigma = 1e-2

# In[2]:

"""Generate a target spectrum
We simulate a target spectrum with a fixed number of spehre's with 
radii sampled from a narrow lognormal distribution
"""
T0 = time.time()
savedir = '../figures/MOO_SUCB/'
if  os.path.exists(savedir):
	shutil.rmtree(savedir)
os.makedirs(savedir)

sim = head.Emulator()

fig, axs = plt.subplots(1,3,figsize=(4*3,4))

sim.make_structure(r_mu=R_mu,r_sigma=R_sigma)
sim.plot_radii(axs[0])
axs[0].set_xlabel('radius')
axs[0].set_ylabel('Probability density')

q, st = sim.get_saxs(n_samples=N_SAS_SAMPLES)
axs[1].loglog(q, st)
plt.setp(axs[1], xlabel='q (1/A)', ylabel='I(q)')

wl, It = sim.get_spectrum(n_samples=N_UVVIS_SAMPLES)
axs[2].plot(wl,It)
plt.setp(axs[2], xlabel=r'$\lambda$ (nm)', ylabel='abs (a.u.)')

fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))

plt.show()


# In[3]:


"""
create a search space as a grid of mu and sigma values of lognormal radii distribution

"""

X = np.linspace(10,30, num=NUM_GRID_PERDIM) 
Y = np.linspace(1e-3,1, num=NUM_GRID_PERDIM)
grid = head.Grid(X,Y)
fig, ax = plt.subplots()
ax.scatter(grid.points[:,0], grid.points[:,1], label='Grid points')
ax.scatter(R_mu, R_sigma, s=100, color='tab:red', label='Target')
ax.set_xlabel(r'$r_{\mu}$')
ax.set_ylabel(r'$r_{\sigma}$')
plt.savefig(savedir + '/dspace.png', bbox_inches='tight')
plt.close()


# In[4]:

def oracle(x):
    """Scoring function at a given input location
    Uses the simulator sim to generate response spectra at a given locations
    and return a similarity score to target spectra
    """
    x_np = x.cpu().numpy()
    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
    q, si = sim.get_saxs(n_samples=N_SAS_SAMPLES)
    wl, Ii = sim.get_spectrum(n_samples=N_UVVIS_SAMPLES)
    dist_sas = euclidean_dist(np.log10(si),np.log10(st))
    dist_uvvis = euclidean_dist(Ii,It)
    return torch.from_numpy(np.asarray([dist_sas, dist_uvvis]))

def batch_oracle(x):
    out = []
    for xi in x.squeeze(1):
        out.append(oracle(xi))
    return torch.stack(out, dim=0).to(**tkwargs)


# In[5]:


problem = lambda s : batch_oracle(s)
ref_point = torch.tensor([0,0]).to(**tkwargs)


# sample initial data
def generate_initial_data(n=6):
    points = torch.from_numpy(grid.points)
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    train_xid = torch.floor(soboleng.draw(n)*len(grid)).to(**tkwargs)
    train_x = points[train_xid.long(),:]
    train_obj = problem(train_x)
    
    return torch.squeeze(train_x).to(**tkwargs), torch.squeeze(train_obj).to(**tkwargs)

N_INIT_SAMPLES = 10
train_x, train_obj = generate_initial_data(n=N_INIT_SAMPLES)
print('Generated %d samples randomly'%N_INIT_SAMPLES, train_x.shape, train_obj.shape)

# In[7]:


# 2. define your model

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(train_x, train_obj)

# 3. Define acqusition function

obj = LinearMCObjective(weights=torch.tensor([0.5, 0.5]).to(**tkwargs))
acq_fun = lambda model: qUpperConfidenceBound(model, beta=0.1, objective=obj)


# 4. define a optimization routine for acqusition function 

def selector(f, q = BATCH_SIZE):
    choices = torch.from_numpy(grid.points).to(**tkwargs)
    new_x, _ = optimize_acqf_discrete(
        acq_function = f,
        q=q,
        choices = choices
    )
    new_obj = problem(new_x)
    return new_x, new_obj


# 5. define the opitmization loop

torch.manual_seed(0)

assert len(grid)>(BATCH_SIZE*N_ITERATIONS + N_INIT_SAMPLES) ,"Not enough samples in the grid"

# run N_ITERATIONS rounds of BayesOpt after the initial random batch
for iteration in range(1, N_ITERATIONS + 1):    
    print('Iteration : %d/%d'%(iteration, N_ITERATIONS))
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition modules using a QMC sampler
    acquisition = acq_fun(model)

    # optimize acquisition functions and get new observations
    new_x, new_obj = selector(acquisition)

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll, model = initialize_model(train_x, train_obj)

    best = train_obj.max(axis=0).values
    print('Best SAS distance : %.2f, Best UVVis distance : %.2f'%(best[0], best[1]))

# Plot paretofront

fig, axes = plt.subplots(1, 1)
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(N_INIT_SAMPLES), torch.arange(1, N_ITERATIONS+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

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
cbar.ax.set_title("Iteration")
plt.savefig(savedir + '/paretofront.png', bbox_inches='tight')
plt.close()

# plot the model
model.eval()
model.likelihood.eval()
XX, YY = grid.mesh
test_x = np.vstack(map(np.ravel, [XX, YY])).T
observed_pred = model.likelihood(model(torch.from_numpy(test_x)))
f1, f2 = observed_pred.mean.detach().numpy()

fig, axs = plt.subplots(1,2,figsize=(5*2, 5))
for ax, f in zip(axs,[f1,f2]):
    sc = ax.contourf(XX,YY,f.reshape(100,100))
    fig.colorbar(sc, ax=ax)
    ax.scatter(train_x[:,0], train_x[:,1], c=batch_number, cmap='bwr')
plt.show()

# plot optimal curves
opt_obj, opt_x = train_obj.max(axis=0)
opt = opt_x.cpu().numpy().squeeze()
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
sim.make_structure(r_mu=opt[0],r_sigma=opt[1])
sim.plot_radii(axs[0])
axs[0].set_xlabel('radius')
axs[0].set_ylabel('Probability density')

q, sopt = sim.get_saxs(n_samples=N_SAS_SAMPLES)
axs[1].loglog(q, sopt, label='Optimal')
axs[1].loglog(q, st, label='Target')
axs[1].legend()
fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))

wl, Iopt = sim.get_spectrum(n_samples=N_UVVIS_SAMPLES)
axs[2].plot(wl,Iopt, label='Optimal')
axs[2].plot(wl,It, label='Target')
axs[1].legend()
plt.savefig(savedir + '/optimums.png', bbox_inches='tight')
plt.close()

print('Best SAS distance : %.2f, Best UVVis distance : %.2f'%(opt_obj.squeeze()[0], opt_obj.squeeze()[1]))
print('Time elapsed %.2f'%(time.time()-T0))




