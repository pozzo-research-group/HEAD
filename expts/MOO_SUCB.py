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
from botorch.acquisition import PosteriorMean
from botorch.acquisition.objective import ScalarizedObjective

N_UVVIS_SAMPLES = 100
N_SAS_SAMPLES = 200
NUM_GRID_PERDIM = 20
BATCH_SIZE = 5
N_ITERATIONS = 4
R_mu = 20
R_sigma = 1e-2
WEIGHTS = [0.5,0.5]

expt = {}
EXPT_ID = 0

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

plt.savefig(savedir + '/targets.png', bbox_inches='tight')
plt.close()

print('Generated targets using the simulators...')

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

print('Generated the design space as a grid')

# In[4]:

def oracle(x):
    """Scoring function at a given input location
    Uses the simulator sim to generate response spectra at a given locations
    and return a similarity score to target spectra
    """
    global EXPT_ID
    x_np = x.cpu().numpy()
    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
    q, si = sim.get_saxs(n_samples=N_SAS_SAMPLES)
    wl, Ii = sim.get_spectrum(n_samples=N_UVVIS_SAMPLES)
    dist_sas = euclidean_dist(np.log10(si),np.log10(st))
    dist_uvvis = euclidean_dist(Ii,It)
    expt[EXPT_ID] = [(q,si,dist_sas),(wl,Ii, dist_uvvis)]
    EXPT_ID += 1
    
    return torch.from_numpy(np.asarray([dist_sas, dist_uvvis]))

def batch_oracle(x):
    out = []
    for xi in x.squeeze(1):
        out.append(oracle(xi))
    return torch.stack(out, dim=0).to(**tkwargs)



problem = lambda s : batch_oracle(s)

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

obj = LinearMCObjective(weights=torch.tensor(WEIGHTS).to(**tkwargs))
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

    # do this to re-train Kernel parameters based on the updated dataset
    mll, model = initialize_model(train_x, train_obj)

    best = train_obj.max(axis=0).values
    print('%d : Best SAS distance : %.2f, Best UVVis distance : %.2f'%(iteration, best[0], best[1]))

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
with torch.no_grad():
	test_x = np.vstack(map(np.ravel, [XX, YY])).T
	test_x = torch.from_numpy(test_x).to(**tkwargs)
	observed_pred = model.likelihood(model(test_x))
	f1, f2 = observed_pred.mean.detach().cpu().numpy()

fig, axs = plt.subplots(1,2,figsize=(5*2, 5))
objective_names = ['SAS', 'UVVis']
for i, (ax, f) in enumerate(zip(axs,[f1,f2])):
    sc = ax.contourf(XX,YY,f.reshape(NUM_GRID_PERDIM,NUM_GRID_PERDIM))
    fig.colorbar(sc, ax=ax)
    ax.scatter(train_x[:,0].cpu().numpy(), train_x[:,1].cpu().numpy(), 
    c=batch_number, cmap='bwr')
    ax.scatter(R_mu,R_sigma, marker='x', color='k')
    ax.set_title(objective_names[i])
plt.savefig(savedir + '/objectives.png', bbox_inches='tight')
plt.close()


# plot optimal curves
objective = ScalarizedObjective(weights=torch.tensor(WEIGHTS).to(**tkwargs))
opt_x, opt_obj = selector(PosteriorMean(model, objective=objective), q=1)
opt_x = opt_x.cpu().numpy().squeeze()

fig, axs = plt.subplots(1,3,figsize=(4*3,4))
line_target = sim.plot_radii(axs[0])
sim.make_structure(r_mu=opt_x[0],r_sigma=opt_x[1])
line_best = sim.plot_radii(axs[0])
axs[0].legend([line_target, line_best],['Target', 'Best'])
axs[0].set_xlabel('radius')
axs[0].set_ylabel('Probability density')

q, sopt = sim.get_saxs(n_samples=N_SAS_SAMPLES)
axs[1].loglog(q, st, label='Target')
axs[1].loglog(q, sopt, label='Optimal')
axs[1].legend()

wl, Iopt = sim.get_spectrum(n_samples=N_UVVIS_SAMPLES)
axs[2].plot(wl,It, label='Target')
axs[2].plot(wl,Iopt, label='Optimal')
axs[2].legend()

fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))
plt.savefig(savedir + '/optimums.png', bbox_inches='tight')
plt.close()

# plot batchwise trace
for b in np.unique(batch_number):
    fig, axs = plt.subplots(1,2,figsize=(5*2, 5))
    fig.suptitle('Batch number %d'%b)
    for i in np.argwhere(batch_number==b).squeeze():
        sas = expt[i][0]
        uvvis = expt[i][1]
        axs[0].loglog(sas[0], sas[1], label='%.2f'%sas[2])
        axs[1].plot(uvvis[0], uvvis[1], label='%.2f'%uvvis[2])
    axs[0].legend()
    axs[1].legend()
    plt.savefig(savedir + '/trace_b%d.png'%b, bbox_inches='tight')
    plt.close()

print('Time elapsed %.2f'%(time.time()-T0))




