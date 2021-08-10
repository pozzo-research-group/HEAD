import os
import torch
import time
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
import pdb
import matplotlib.pyplot as plt
import head
import numpy as np
from head.metrics import euclidean_dist

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.utils.transforms import unnormalize


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

torch.manual_seed(0)

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


savedir = '../figures/MOO/'
if  os.path.exists(savedir):
    shutil.rmtree(savedir)
os.makedirs(savedir)

"""Generate a target spectrum
We simulate a target spectrum with a fixed number of spehre's with 
radii sampled from a narrow lognormal distribution
"""

sim = head.Emulator()
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
R_mu = 1
R_sigma = 1e-2
sim.make_structure(r_mu=R_mu,r_sigma=R_sigma)
sim.plot_structure2d(ax=axs[0])

q, st = sim.get_saxs()
axs[1].loglog(q, st)
fig.suptitle('r = '+','.join('%.2f'%i for i in sim.step*sim.radii))

wl, It = sim.get_spectrum()
axs[2].plot(wl,It)
plt.savefig(savedir+'target.png', dpi=500, bbox_inches='tight')


"""
create a search space as a grid of mu and sigma values of lognormal radii distribution
"""

NUM = 5
X = np.linspace(0.75,1.25, num=NUM) 
Y = np.linspace(1e-3,1e-1, num=NUM)
grid = head.Grid(X,Y)
fig, ax = plt.subplots()
ax.scatter(grid.points[:,0], grid.points[:,1], label='Grid points')
ax.scatter(R_mu, R_sigma, s=100, color='tab:red', label='Target')
ax.set_xlabel(r'$r_{\mu}$')
ax.set_ylabel(r'$r_{\sigma}$')
plt.savefig(savedir+'dspace.png', dpi=500, bbox_inches='tight')


def oracle(x):
    """Scoring function between two spectra
    Given two spectra in si, st compute a score between them.
    Note that here si, st are represented using the class `head.UVVis`.
    If you are using arrays, you'd have to make necessary transformations.
    """
    x_np = x.numpy()
    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
    q, si = sim.get_saxs()
    wl, Ii = sim.get_spectrum()
    dist_sas = euclidean_dist(np.log10(si),np.log10(st))
    dist_uvvis = euclidean_dist(Ii,It)
    return torch.from_numpy(np.asarray([-dist_sas, -dist_uvvis]))

def batch_oracle(x):
    out = []
    for xi in x.squeeze(1):
        out.append(oracle(xi))
    return torch.stack(out, dim=0)


problem = lambda s : batch_oracle(s).to(**tkwargs)
ref_point = torch.tensor([0,0]).to(**tkwargs)

def generate_initial_data(n=6):
    points = torch.from_numpy(grid.points)
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    train_xid = torch.floor(soboleng.draw(n)*len(grid)).to(**tkwargs)
    train_x = points[train_xid.long(),:]
    train_obj = problem(train_x)
    
    return torch.squeeze(train_x), torch.squeeze(train_obj)

train_x, train_obj = generate_initial_data(n=6)

# 2. define your model

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(train_x, train_obj)

partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)
MC_SAMPLES = 128
sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

acq_fun = lambda model: qExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point.tolist(),  # use known reference point 
    partitioning=partitioning,
    sampler=sampler,
)

# 4. define a optimization routine for acqusition function 

BATCH_SIZE = 4 

def selector(f, q = BATCH_SIZE):
    choices = torch.from_numpy(grid.points)
    new_x, _ = optimize_acqf_discrete(
        acq_function = f,
        q=q,
        choices = choices
    )
    new_obj = problem(new_x)
    return new_x, new_obj

# 5. define the opitmization loop

N_ITERATIONS = 25

verbose = False
hv = Hypervolume(ref_point=ref_point)

hvs_all = []
hvs = []

# compute pareto front
pareto_mask = is_non_dominated(train_obj)
pareto_y = train_obj[pareto_mask]

# compute hypervolume
volume = hv.compute(pareto_y)
hvs.append(volume)

# run N_ITERATIONS rounds of BayesOpt after the initial random batch
for iteration in range(1, N_ITERATIONS + 1):    
    
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition modules using a QMC sampler
    acquisition = acq_fun(model)

    # optimize acquisition functions and get new observations
    new_x, new_obj = selector(acquisition)

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    # compute pareto front
    pareto_mask = is_non_dominated(train_obj)
    pareto_y = train_obj[pareto_mask]
    
    # compute hypervolume
    volume = hv.compute(pareto_y)
    hvs.append(volume)

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll, model = initialize_model(train_x, train_obj)

    print(".", end="")

    hvs_all.append(hvs)


fig = plt.figure(figsize=(2*5,5))
axes = fig.add_subplot(1,2,1)

cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(6), torch.arange(1, N_ITERATIONS+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

sc = axes.scatter(train_obj[:, 0].cpu().numpy(), train_obj[:,1].cpu().numpy(), 
    c=batch_number, alpha=0.8,
)
axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm =  ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

iters = np.arange(N_ITERATIONS + 1) * BATCH_SIZE
ax = fig.add_subplot(1,2,2)
ax.plot(
    iters, np.asarray(hvs_all).mean(axis=0), linewidth=1.5,
)

ax.set(xlabel='number of observations (beyond initial points)', ylabel='Hypervolume')
plt.savefig(savedir+'evaluations.png', dpi=500, bbox_inches='tight')

opt_x, opt_obj = selector(acquisition,q=1)
opt = opt_x.numpy().squeeze()
fig, axs = plt.subplots(1,3,figsize=(4*3,4))
sim.make_structure(r_mu=opt[0],r_sigma=opt[1])
sim.plot_structure2d(ax=axs[0])

q, sopt = sim.get_saxs()
axs[1].loglog(q, sopt, label='Optimal')
axs[1].loglog(q, st, label='Target')
fig.suptitle('r = '+','.join('%.2f'%i for i in sim.step*sim.radii))

wl, Iopt = sim.get_spectrum()
axs[2].plot(wl,Iopt, label='Optimal')
axs[2].plot(wl,It, label='Target')
plt.savefig(savedir+'optimum.png', dpi=500, bbox_inches='tight')




