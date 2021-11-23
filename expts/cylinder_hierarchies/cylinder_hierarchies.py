#!/usr/bin/env python
# coding: utf-8

import os, shutil
import torch
import time
import pdb
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 15,
                     "savefig.dpi": 400,
                     "savefig.bbox": 'tight'
                    }
                   )
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
import head
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import  qExpectedImprovement
from matplotlib.cm import ScalarMappable
from scipy.spatial import distance
import pdb
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import PosteriorMean
from matplotlib import cm
import matplotlib
from head import EmulatorSingleParticle
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.functions import L2Space, SinfSpace, ProbabilityDistributions, SRVF
from geomstats import visualization as viz
import geomstats.visualization as viz
from geomstats.geometry.hyperboloid import Hyperboloid

N_SAMPLES = 100
BATCH_SIZE = 4
N_ITERATIONS = 15
NUM_RESTARTS = 64 
RAW_SAMPLES = 1024
N_INIT_SAMPLES = 2
NUM_DIPOLES = 100

TARGET = [5,45]
VERBOSE = True
savedir = './'
metric = 'srvf'

print('Using the %s metric'%metric)

class InputTransform:
    def __init__(self, bounds):
        self.min = bounds.min(axis=0).values
        self.range = bounds.max(axis=0).values-self.min
        
    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min
 
# define search space
param_r = [2,20]
param_theta = [0,90]
bounds_params = torch.tensor((param_r, param_theta)).T.to(**tkwargs)
inp = InputTransform(bounds_params)
xt = inp.transform(torch.tensor(TARGET)).reshape(1,2).numpy()

bounds = torch.tensor(([1e-3,1], [1e-3,1])).T.to(**tkwargs) 
   
sim = EmulatorSingleParticle(verbose=False)
fig, ax = plt.subplots()

lambda_, yt = sim.get_uvvis(radius=TARGET[0], length=15, theta=TARGET[1],
 num_dipoles=NUM_DIPOLES)
ax.plot(lambda_, yt)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Optical extinction efficiency')
plt.savefig(savedir+'target.png')
 
def draw_random_batch(n_samples=6):
    train_x = draw_sobol_samples(
        bounds=bounds,n=1, q=n_samples,
        seed=torch.randint(2021, (1,)).item()).squeeze(0)
    return train_x

random_x = draw_random_batch(n_samples=N_INIT_SAMPLES)


H2 = PoincareHalfSpace(2)
Rn = Euclidean(N_SAMPLES)
R2 = Euclidean(len(TARGET))
L2 = L2Space(lambda_)
Sinf = SinfSpace(lambda_)
PD = ProbabilityDistributions(lambda_)
srvf = SRVF(lambda_)


if metric=='L2':
    d = lambda xi,yi : -L2.metric.dist(yi, yt)
elif metric=='Rn':
    d = lambda xi,yi : -float(Rn.metric.dist(yi, yt))
elif metric=='H2':
    xt = np.asarray(TARGET).reshape(1,2)
    d = lambda xi,yi : -float(H2.metric.dist(xi.reshape(1,2),xt))
elif metric=='R2':
    xt = np.asarray(TARGET).reshape(1,2)
    d = lambda xi,yi : -float(R2.metric.dist(xi.reshape(1,2),xt))  
elif metric=='Sinf':
    d = lambda xi,yi : -Sinf.metric.dist(yi, yt)
elif metric=='PD':
    def d(xi,yi):
        yi_p = PD.projection(yi)
        yt_p = PD.projection(yt)
        
        return -PD.metric.dist(yi_p, yt_p)
elif metric=='srvf':
    d = lambda xi,yi : -srvf.metric.dist(yi, yt)     
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
        x = inp.inverse(x)
        x_np = x.cpu().numpy()
        _, yi = sim.get_uvvis(radius=x_np[0], length=15, theta=x_np[1],
            num_dipoles = NUM_DIPOLES)
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
for iteration in range(1, N_ITERATIONS + 1):
    mll, model = initialize_model(train_x, train_obj)
 
    if VERBOSE:
        print('Iteration : %d/%d'%(iteration, N_ITERATIONS))
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition module
    #acquisition = qUpperConfidenceBound(model, beta=0.1)
    best_f = train_obj.max(axis=0).values
    acquisition = qExpectedImprovement(model, best_f = best_f)
    
    # optimize acquisition functions and get new observations
    new_x, new_obj = selector(acquisition, oracle)
    if VERBOSE:
        for i in range(BATCH_SIZE):
            print('%d\t%s\t%s'%(i, new_x[i,...].numpy(), 
                                new_obj[i,...].numpy()))

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

expt = oracle.expt
train_x = inp.inverse(train_x.cpu().numpy())
xt = inp.inverse(xt)

print('Plotting the distance between best and target in search space ...')
batch_number = torch.cat(
    [torch.zeros(N_INIT_SAMPLES), 
     torch.arange(1, N_ITERATIONS+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

def plot_best_trace(ax, train_x, train_obj, target):
    proximities = distance.cdist(train_x, xt)
    trace = np.asarray([min(proximities[batch_number<=b]) for b in np.unique(batch_number)])
    return ax.plot(np.arange(N_ITERATIONS+1),trace)
fig, ax = plt.subplots()
ax.axhline(0, label='Optimal', ls='--', lw='2.0', c='k')
plot_best_trace(ax, train_x, train_obj, TARGET)
ax.set_xlabel('Batch number')
ax.set_ylabel(r'$||x-x_{t}||_{2}$')
plt.savefig('design_space_trace.png')

print('Plotting the optimization trace ...')
fig, axs = plt.subplots(1,2,figsize=(5*2,5))
batches = np.unique(batch_number)
best_ind = [np.argmax(train_obj.numpy()[batch_number<=b]) for b in batches]
best_loc = train_x[best_ind].numpy()

ax = axs[0]
ax.plot(lambda_, yt, 'k--', lw=2.0)
cmap = cm.get_cmap('coolwarm')
norm = matplotlib.colors.Normalize(vmin=0, vmax = len(batches))
for i, bind in enumerate(best_ind):
    wl, ext, _ = expt[bind]
    ax.plot(wl, ext, color=cmap(norm(i)))
    
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$\phi(x)$', fontsize=14)
cax = plt.axes([0.95, 0.2, 0.01, 0.6])        
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.ax.set_ylabel('Batch number/ Time', rotation=270)
cbar.ax.get_yaxis().labelpad = 15

ax = axs[1]
viz.plot(best_loc,c=batches,cmap='coolwarm', 
              space='H2_poincare_half_plane', ax=ax)
viz.plot(np.asarray(TARGET).reshape(1,2), space='H2_poincare_half_plane', ax=ax,  
         marker='*', s=100, facecolors='none', color='k', lw=2.0)

ax.set_xlabel(r'$r$', fontsize=20)
ax.set_ylabel(r'$\theta$', fontsize=20)
ax.set_xlim(bounds_params.numpy()[:,0] + np.asarray([-0.5,0.5]))
ax.set_ylim(bounds_params.numpy()[:,1] + np.asarray([-0.5,0.5]))
    
plt.savefig(savedir+'/trace_%s.png'%metric)


# plot each batch selection
print('Plotting all the collected samples ...')
fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
ax = axs[0]
cmap = cm.get_cmap('coolwarm')
norm = matplotlib.colors.Normalize(vmin=0, vmax = len(batches))
for b in batches:
    ax.scatter(train_x[batch_number==b,0], train_x[batch_number==b,1], 
               color=cmap(norm(b)))
ax.set_xlim(bounds_params.numpy()[:,0] + np.asarray([-0.1,0.1]))
ax.set_ylim(bounds_params.numpy()[:,1] + np.asarray([-0.1,0.1]))
ax.scatter(xt[0][0], xt[0][1], marker='o', 
           s=100, facecolors='none', color='k', lw=2.0)

ax.set_xlabel(r'$r$', fontsize=14)
ax.set_ylabel(r'$\theta$', fontsize=14)

ax = axs[1]
ax.plot(lambda_, yt, 'k--', lw=2.0)
for b in batches:
    loc = np.where(batch_number==b)[0]
    for l in loc:
        xi, yi, _ = expt[l]
        ax.plot(xi, yi, color=cmap(norm(b)))
ax.set_xlim([min(lambda_),max(lambda_)])
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\phi(x, \lambda)$', fontsize=14)
cax = plt.axes([0.95, 0.2, 0.01, 0.6])        
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.ax.set_ylabel('Batch number', rotation=270)
cbar.ax.get_yaxis().labelpad = 15

plt.savefig('all_trainx.png')

print('Plotting the Poincaré Ball Representation ...')
Hb = Hyperboloid(2)

fig, ax = plt.subplots()
disk = viz.PoincareDisk()
circle = plt.Circle((0, 0), radius=1.0, color="black", fill=False)
ax.add_patch(circle)
pt_minkowski = np.hstack(((batches+1).reshape(N_ITERATIONS+1,1), best_loc))
pt_hb = Hb.projection(pt_minkowski)
disk.add_points(pt_hb)
disk.draw(ax,c=batches,cmap='coolwarm')
ax.axis('equal')
ax.axis('off')
plt.savefig('poincare_ball.png')
