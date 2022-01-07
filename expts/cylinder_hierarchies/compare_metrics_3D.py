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
                     "savefig.bbox": 'tight',
                     'text.latex.preamble': r'\usepackage{amsfonts}'
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
N_ITERATIONS = 20
NUM_RESTARTS = 64 
RAW_SAMPLES = 1024
N_INIT_SAMPLES = 2
NUM_DIPOLES = 100
N_REPEAT = 5

TARGET = [5,15,45]
VERBOSE = False
savedir = './'

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
param_length = [2,20]
bounds_params = torch.tensor((param_r, param_length, param_theta)).T.to(**tkwargs)
inp = InputTransform(bounds_params)
xt = inp.transform(torch.tensor(TARGET)).reshape(1,len(TARGET)).numpy()

bounds = torch.tensor(([1e-3,1],[1e-3,1],[1e-3,1])).T.to(**tkwargs) 
   
sim = EmulatorSingleParticle(verbose=False)
lambda_, yt = sim.get_uvvis(radius=TARGET[0], length=TARGET[1], theta=TARGET[2],
 num_dipoles=NUM_DIPOLES)

def draw_random_batch(n_samples=6):
    train_x = draw_sobol_samples(
        bounds=bounds,n=1, q=n_samples,
        seed=torch.randint(2021, (1,)).item()).squeeze(0)
    return train_x

random_x = draw_random_batch(n_samples=N_INIT_SAMPLES)


Rn = Euclidean(N_SAMPLES)
L2 = L2Space(lambda_)
PD = ProbabilityDistributions(lambda_)
srvf = SRVF(lambda_)

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
        _, yi = sim.get_uvvis(radius=x_np[0], length=x_np[1], theta=x_np[2],
            num_dipoles = NUM_DIPOLES)
        dist = self.metric(x_np, yi)

        self.expt[self.expt_id] = [lambda_, yi, dist]
    
        return torch.tensor([dist])
    
    def batch_evaluate(self, x):
        if VERBOSE:
            print('Current experiment id : ', self.expt_id)
        out = []
        for xi in x.squeeze(1):
            out.append(self.evaluate(xi))
            self.expt_id += 1
        return torch.stack(out, dim=0).to(**tkwargs)
    

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


def run(metric, xt):
    if metric=='L2':
        d = lambda xi,yi : -L2.metric.dist(yi, yt)
    elif metric=='Rn':
        d = lambda xi,yi : -float(Rn.metric.dist(yi, yt))
    elif metric=='srvf':
        d = lambda xi,yi : -srvf.metric.dist(yi, yt)     
    else:
        raise NotImplementedError('Metric %s is not implemented'%metric)

    oracle = Oracle(d)
    
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
        mll, model = initialize_model(train_x, train_obj)
     
        if VERBOSE:
            print('Iteration : %d/%d'%(iteration, N_ITERATIONS))
        # fit the models
        fit_gpytorch_model(mll)

        # define the acquisition module
        best_f = train_obj.max(axis=0).values
        acquisition = qExpectedImprovement(model, best_f = 0.0)
        
        # optimize acquisition functions and get new observations
        if iteration>0:
            new_x, new_obj = selector(acquisition, oracle)
            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
        
        # obtain the current best from the model using posterior
        opt_x, opt_obj = selector(PosteriorMean(model), oracle, q=1)
        best_loc.append(opt_x)
        best_ind.append(oracle.expt_id-1)
        
        if VERBOSE:
            for i in range(BATCH_SIZE):
                print('%d\t%s\t%s'%(i, new_x[i,...].numpy(), 
                                    new_obj[i,...].numpy()))

    expt = oracle.expt
    best_loc = inp.inverse(torch.cat(best_loc).numpy())
    train_x = inp.inverse(train_x.cpu().numpy())
    xt = inp.inverse(xt)
        
    proximities = distance.cdist(best_loc, xt)
    
    return proximities
  
# Perform the experiment
METRICS = ['Rn', 'srvf']  
proximities = np.zeros((len(METRICS), N_REPEAT,N_ITERATIONS + 1))    
for j in range(N_REPEAT):
    torch.seed()
    random_x = draw_random_batch(n_samples=N_INIT_SAMPLES)
    for i,metric in enumerate(METRICS):
        proximities[i,j,:] = run(metric, xt).squeeze()
        print('Repeat %d\tMetric : %s\tFinal distance: %.2f'%(j,metric,proximities[i,j,:][-1]))
np.save('proximities.npy', proximities)   

fig, axs = plt.subplots(1,len(METRICS), figsize=(4*len(METRICS), 4), sharey=True, sharex=True) 
labels = [r'$\mathbb{R}^n$', 'SRVF']    
for i,_ in enumerate(METRICS):
    ax = axs[i]
    y_mean = proximities.mean(axis=1)[i,...]
    y_std = proximities.std(axis=1)[i,...]
    ax.plot(np.arange(N_ITERATIONS+1), y_mean, lw=2.0)
    ax.fill_between(np.arange(N_ITERATIONS+1), y_mean-y_std, y_mean+y_std, alpha=0.2)
    ax.set_title(labels[i])
    ax.axhline(0, ls='--', lw='2.0', c='k')  

axs[0].set_ylabel(r'$||x-x_{t}||_{2}$')
fig.supxlabel('Batch number', y=-0.05)
plt.savefig('metric_compare.png')
plt.show()

