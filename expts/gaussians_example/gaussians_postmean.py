#!/usr/bin/env python
# coding: utf-8

import os, shutil, time, pdb
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 15,
                     "savefig.dpi": 400,
                     "savefig.bbox": 'tight',
                     'text.latex.preamble': r'\usepackage{amsfonts}'
                    }
                   )

import torch
torch.seed()

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import  qExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import PosteriorMean

from matplotlib import cm
import matplotlib
from matplotlib.cm import ScalarMappable

import geomstats.backend as gs
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.functions import L2Space, SRVF
import geomstats.visualization as viz

from scipy.spatial import distance

import fdasrsf as fs

import head

N_SAMPLES = 100
BATCH_SIZE = 4
N_ITERATIONS = 24
NUM_RESTARTS = 64 
RAW_SAMPLES = 1024
N_INIT_SAMPLES = 4

TARGET = [-2,0.5]
VERBOSE = False
savedir = './'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
metric = 'srvf'
print('Using the %s metric'%metric)

lambda_ = np.linspace(-5,5,num=N_SAMPLES)
def gaussian(mu,sig):
    scale = 1/(np.sqrt(2*np.pi)*sig)
    return scale*np.exp(-np.power(lambda_ - mu, 2.) / (2 * np.power(sig, 2.)))

yt = gaussian(*TARGET)

fig, ax = plt.subplots()
ax.plot(lambda_, yt)
plt.savefig(savedir+'target.png')

# define search space
param_mu = [-10,10]
param_sig = [0.1,3.5]
bounds = torch.tensor((param_mu, param_sig)).T.to(**tkwargs)


def draw_random_batch(n_samples=6):
    train_x = draw_sobol_samples(
        bounds=bounds,n=1, q=n_samples, 
        seed=torch.randint(2021, (1,)).item()
    ).squeeze(0)
    return train_x

random_x = draw_random_batch(n_samples=N_INIT_SAMPLES)


# Metric selection
H2 = PoincareHalfSpace(2)
Rn = Euclidean(N_SAMPLES)
L2 = L2Space(lambda_)
srvf = SRVF(lambda_)

if metric=='L2':
    d = lambda xi,yi : -L2.metric.dist(yi, yt)
elif metric=='Rn':
    d = lambda xi,yi : -float(Rn.metric.dist(yi, yt))
elif metric=='srvf':
    d = lambda xi,yi : -srvf.metric.dist(yi, yt) 
elif metric=='ap':
    def d(xi, yi):
        curves = np.zeros((len(yt), 2))
        curves[...,0] = yt
        curves[...,1] = yi
        obj = fs.fdawarp(curves, lambda_)
        obj.srsf_align(parallel=True)
        dp = fs.efda_distance_phase(obj.qn[...,0], obj.qn[...,1])
        da = fs.efda_distance(obj.qn[...,0], obj.qn[...,1])
        
        return -(da+dp)    
else:
    raise NotImplementedError('Metric %s is not implemented'%metric)

class Oracle:
    def __init__(self, metric):
        self.metric = metric
        self.expt_id = 0
        self.expt = {}
        
    def evaluate(self, x):
        """Scoring function at a given input location
        Uses the simulator sim to generate response spectra at a given locations
        and return a similarity score to target spectra
        """
        x_np = x.cpu().numpy()
        yi = gaussian(x_np[0],x_np[1])
        dist = self.metric(x_np, yi)

        self.expt[self.expt_id] = [lambda_, yi, dist]
    
        return torch.tensor([dist])
    
    def batch_evaluate(self, x):
        print('Current experiment id : ', self.expt_id)
        out = []
        for xi in x.squeeze(1):
            out.append(self.evaluate(xi))
            self.expt_id += 1
        return torch.stack(out, dim=0).to(**tkwargs)
    
oracle = Oracle(d)

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def selector(f,oracle, q = BATCH_SIZE):
    new_x, _ = optimize_acqf(
        acq_function=f,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES, 
        sequential=False,
    )
    new_obj = oracle.batch_evaluate(new_x)
    return new_x, new_obj


# ## Main optimization loop
# Evaluate the randomly selected points
train_x = random_x.clone()
train_obj = oracle.batch_evaluate(train_x)
if VERBOSE:
    print('Generated %d samples randomly'%N_INIT_SAMPLES, train_x.shape, train_obj.shape)
    for i in range(N_INIT_SAMPLES):
        print('%d\t%s\t%s'%(i, train_x[i,...].numpy(), train_obj[i,...].numpy()))
        
# run N_ITERATIONS rounds of BayesOpt after the initial random batch
if VERBOSE:
    print('Sampled ID \t Locations \t Objectives')
    
best_ind = []
best_loc = []
for iteration in range(N_ITERATIONS + 1): 
    # re-initialize
    mll, model = initialize_model(train_x, train_obj)
    if VERBOSE:
        print('Iteration : %d/%d'%(iteration, N_ITERATIONS))
    # fit the models
    fit_gpytorch_model(mll)

    # optimize acquisition functions and get new observations
    if iteration>0:
        # define the acquisition module
        best_f = train_obj.max(axis=0).values
        acquisition = qExpectedImprovement(model, best_f = 0.0)
        new_x, new_obj = selector(acquisition, oracle)
        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        if VERBOSE:
            for i in range(BATCH_SIZE):
                print('%d\t%s\t%s'%(i, new_x[i,...].numpy(), 
                                    new_obj[i,...].numpy()))
    
    # obtain the current best from the model using posterior
    opt_x, opt_obj = selector(PosteriorMean(model), oracle, q=1)
    best_loc.append(opt_x)
    best_ind.append(oracle.expt_id-1)


expt = oracle.expt
best_loc = torch.cat(best_loc).numpy()
xt = np.asarray(TARGET).reshape(1,2)

print('Plotting the distance between target and design space points ...')

fig, ax = plt.subplots()
ax.axhline(0, label='Optimal', ls='--', lw='2.0', c='k')
trace = distance.cdist(best_loc, xt)
ax.plot(np.arange(N_ITERATIONS+1),trace)
ax.set_xlabel('Batch number')
ax.set_ylabel(r'$||x-x_{t}||_{2}$')
plt.savefig(savedir+'/trace_design_space_%s.png'%metric)

print('Plotting the trace in spectra and design space ...')
fig, axs = plt.subplots(1,2,figsize=(5*2,5))
batches = np.arange(N_ITERATIONS+1)

ax = axs[0]
ax.plot(lambda_, yt, 'k--', lw=2.0)
cmap = cm.get_cmap('coolwarm')
norm = matplotlib.colors.Normalize(vmin=0, vmax = N_ITERATIONS+1)
for i, bind in enumerate(best_ind):
    wl, ext, _ = expt[bind]
    ax.plot(wl, ext, color=cmap(norm(i)))
ax.set_xlim([-5,5])
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\phi(\lambda;x)$', fontsize=14)
cax = plt.axes([0.95, 0.2, 0.01, 0.6])        
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.ax.set_ylabel('Batch number/ Time', rotation=270)
cbar.ax.get_yaxis().labelpad = 15

ax = axs[1]
ax.scatter(best_loc[:,0], best_loc[:,1],c=batches,
    cmap='coolwarm', label='BO trace')
ax.scatter(TARGET[0], TARGET[1], marker='*', s=100, 
    facecolors='none', color='k', lw=2.0, label='Target')

x0 = best_loc[0,:]
n_points = 20
t = np.linspace(0, 1, n_points)
geodesic = H2.metric.geodesic(x0, xt)(t)
ax.plot(geodesic[:,0], geodesic[:,1], lw=2.0, c='k', label='True geodesic')

ax.set_xlabel(r'$\mu$', fontsize=20)
ax.set_ylabel(r'$\sigma$', fontsize=20)
ax.set_xlim(bounds.numpy()[:,0] + np.asarray([-0.1,0.1]))
ax.set_ylim(bounds.numpy()[:,1] + np.asarray([-0.1,0.1]))
ax.legend()
plt.savefig(savedir+'/trace_%s.png'%metric)

with torch.no_grad():
    num_grid_spacing = 20
    mu_grid = np.linspace(*bounds[:,0].numpy(), num=num_grid_spacing)
    sig_grid = np.linspace(*bounds[:,1].numpy(), num=num_grid_spacing)
    test_x = head.Grid(mu_grid, sig_grid).points
    posterior = model.posterior(torch.tensor(test_x).to(**tkwargs))
    posterior_mean = posterior.mean.cpu().numpy()
    XX, YY = np.meshgrid(mu_grid, sig_grid)
    Z = posterior_mean.reshape(num_grid_spacing,num_grid_spacing)
    fig, ax = plt.subplots()
    ax.contourf(XX, YY, Z, cmap=cm.Blues)
    ax.scatter(TARGET[0], TARGET[1], marker='*', s=200,color='tab:red', lw=2.0,fc='none')
    ax.scatter(opt_x[0][0], opt_x[0][1], marker='*', s=200,lw=2.0,fc='none',color='k')

    plt.savefig(savedir+'/model_%s.png'%metric)
    
    
    