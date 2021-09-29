#!/usr/bin/env python
# coding: utf-8

import os, shutil
import torch
import time
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
import pdb
import matplotlib.pyplot as plt

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
from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from matplotlib.cm import ScalarMappable
from head import SymmetricMatrices
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import PosteriorMean
from matplotlib import cm
from matplotlib.colors import Normalize

torch.manual_seed(2021)
import pdb, logging, pickle

N_UVVIS_SAMPLES = 15
N_SAS_SAMPLES = 100
NUM_GRID_PERDIM = 100
BATCH_SIZE = 4
N_ITERATIONS = 5
NUM_RESTARTS = 64 
RAW_SAMPLES = 1024
N_INIT_SAMPLES = BATCH_SIZE

R_mu = 20
R_sigma = 1e-2
SHAPE_PARAM = 0.2
SPECTRA = 'saxs'


class TestShapeMatchBO:
    def __init__(self, metric):
        """Evaluate performance of a metric in a single run
        
            metric  :   A function that takes SIX arrays of shape (n,) and return a scalar distance
                        (e.g.: scipy.spatial.distance.euclidean wrapped to be used with two sets of point clouds)
        """
        self.metric = metric
        self.sim = head.EmulatorMultiShape()
        self.sim.make_structure(r_mu=R_mu,r_sigma=R_sigma)

        if SPECTRA=='saxs':
            self.xt, self.yt = self.sim.get_saxs(shape_param = SHAPE_PARAM, n_samples=N_SAS_SAMPLES)
        elif SPECTRA=='uvvis':
            self.xt, self.yt = self.sim.get_uvvis(shape_param = 0.67, n_samples=N_UVVIS_SAMPLES)

        # define search space
        self.r_mu = [5,50]
        self.r_sigma = [1e-4,1]
        self.shape_param = [0,1]
        self.bounds = torch.tensor((self.r_mu, self.r_sigma, self.shape_param)).T.to(**tkwargs)
        self.EXPT_ID = 0
        self.expt = {}
        self.target = np.array([R_mu, R_sigma, SHAPE_PARAM])

    def oracle(self, x):
        """Scoring function at a given input location
        Uses the simulator sim to generate response spectra at a given locations
        and return a similarity score to target spectra
        """
        x_np = x.cpu().numpy()    
        self.sim.make_structure(r_mu=x_np[0],
                       r_sigma=x_np[1])
        if SPECTRA=='saxs':
            xi, yi = self.sim.get_saxs(shape_param = x_np[2], 
                                  n_samples=N_SAS_SAMPLES)
            dist = self.metric(np.log10(xi), np.log10(yi), 
                np.log10(self.xt), np.log10(self.yt), x_np, self.target)
        elif SPECTRA=='uvvis':
            xi, yi = self.sim.get_uvvis(shape_param = x_np[2], 
                                   n_samples=N_UVVIS_SAMPLES)
            dist = self.metric(xi, yi, self.xt, self.yt, x_np, self.target)
            
        self.expt[self.EXPT_ID] = [xi, yi, dist]
        self.EXPT_ID += 1

        return torch.tensor([dist])
    
    def batch_oracle(self,x):
        out = []
        for xi in x.squeeze(1):
            out.append(self.oracle(xi))
        return torch.stack(out, dim=0).to(**tkwargs)

    def draw_random_batch(self,n_samples=6):
        train_x = draw_sobol_samples(
            bounds=self.bounds,n=1, q=n_samples, 
            seed=torch.randint(2021, (1,)).item()
        ).squeeze(0)
        train_obj = self.batch_oracle(train_x)
        return train_x, train_obj

    def initialize_model(self,train_x, train_obj):
        # define models for objective and constraint
        model = SingleTaskGP(train_x, train_obj, 
        outcome_transform=Standardize(m=train_obj.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def selector(self,f, q = BATCH_SIZE):
        new_x, _ = optimize_acqf(
            acq_function=f,
            bounds=self.bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES, 
            sequential=False,
        )
        new_obj = self.batch_oracle(new_x)
        return new_x, new_obj

    def run(self):
        # run N_ITERATIONS rounds of BayesOpt after the initial random batch
        self.train_x, self.train_obj = self.draw_random_batch(n_samples=N_INIT_SAMPLES)
        mll, model = self.initialize_model(self.train_x, self.train_obj)
        
        for iteration in range(1, N_ITERATIONS + 1):    
            # fit the models
            logging.info('Iteration : %d/%d'%(iteration, N_ITERATIONS))
            fit_gpytorch_model(mll)

            # define the acquisition module
            acquisition = qExpectedImprovement(model, best_f = self.train_obj.max())
            # optimize acquisition functions and get new observations
            new_x, new_obj = self.selector(acquisition)

            # update training points
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_obj = torch.cat([self.train_obj, new_obj])

            # re-initialize
            mll, model = self.initialize_model(self.train_x, self.train_obj)
            
        self.model = model
        self.mll = mll
        self.batch_number = torch.cat(
            [torch.zeros(N_INIT_SAMPLES), 
             torch.arange(1, N_ITERATIONS+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
        ).numpy()
        return self
        
    def plot(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
            
        all_scores = [-v[2] for _, v in self.expt.items()]


        plot_scores = []
        for b in np.unique(self.batch_number):
            scores = np.asarray(all_scores)[np.argwhere(self.batch_number==b)]
            mu, std = scores.mean(), scores.std()
            plot_scores.append([mu, mu+std, mu-std])
            
        plot_scores = np.asarray(plot_scores)
        #ax.fill_between(np.unique(self.batch_number), y1=plot_scores[:,1], y2=plot_scores[:,2], alpha=0.5)
        ax.plot(np.unique(self.batch_number), plot_scores[:,0])
        
        return 
        
    def get_best_from_surrogate(self):
        opt_x, opt_obj = self.selector(PosteriorMean(self.model), q=1)
        opt_x = opt_x.cpu().numpy().squeeze()
        
        return opt_x
        
    def save(self, name=None):
        need_keys = ['xt', 'yt', 'r_mu', 'r_sigma', 'shape_param', 'bounds', 'expt', 
             'target', 'train_x', 'train_obj', 'batch_number']
        out = {}
        for nk in need_keys:
            out[nk] = self.__dict__[nk]
        if name is not None:    
            with open(name+'.pkl', 'wb') as handle:
                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            return out

