import logging
import torch
import pdb
import os
import shutil
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import  qExpectedImprovement
from matplotlib.cm import ScalarMappable
from scipy.spatial import distance
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import PosteriorMean
from geomstats.geometry.euclidean import Euclidean
import pandas as pd 
import pickle

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

 
class InputTransform:
    def __init__(self, bounds):
        self.min = bounds.min(axis=0).values
        self.range = bounds.max(axis=0).values-self.min
        
    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min
 
class Optimizer:
    """Bayesian optimizer for spectral data 
    
    This class is used to perform Bayesian optimization for target spectra 
    using the opentrons robot
    
    Params:
    -------
        xt          :   Target spectra of domain 
                        (n_samples, )
        yt          :   Target spectra of shape 
                        (n_samples, )
        bounds      :   Bounds for the design space 
                        (list of the length n_dim)   
        
        savedir     :   Directory location to save output from the Optimizer 
                        (default, '../')
        batch_size  :   Batch size used for optimization 
                        (default, 8)
        hyperplane  :   Boolean variable to specify whether the desgin space 
                        is a hyperplane 
                        (default, False)
        metric      :   Distance measure used to compute similarity 
                        beteween spectra 
                        (default, Euclidean)
                        This is a function that should take two spectra 
                        of shape (n_samples, ) and return a scalar as a float
    """
    
    def __init__(self, xt, yt, bounds, 
        savedir='../', 
        batch_size=8, 
        hyperplane=False,
        metric = None
        ):

        self.xt = xt 
        self.yt = yt
        
        bounds_params = torch.tensor(bounds).T.to(**tkwargs)
        self.inp = InputTransform(bounds_params)
        self.bounds = torch.tensor(([[1e-3,1] for _ in range(len(bounds))])).T.to(**tkwargs)
        
        self.hyperplane = hyperplane
        self.metric_function = metric
        self.Rn = Euclidean(len(self.xt)) 
        
        if self.hyperplane:
            indices = torch.arange(self.bounds.shape[0], dtype=torch.long, device=tkwargs['device'])
            coeffs = torch.ones(self.bounds.shape[0]).to(**tkwargs)
            self.constraints = [(indices, coeffs, 1.0)]
        else:
            self.constraints = None
        
        self.savedir = savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.batch_size = batch_size
        self.expt = {}
        self.expt_id = 0
        self.train_x = torch.tensor([]).to(**tkwargs)
        self.train_obj = torch.tensor([]).to(**tkwargs)
        
        self.iteration = 0
        self.suggest_next()
        self.new_obj = torch.tensor([]).to(**tkwargs)
        self.best_loc = []

    def metric(self, yi):
        if self.metric_function is None:
            return -float(self.Rn.metric.dist(yi, self.yt))
        else:
            return self.metric_function(yi[0], self.yt)
    
    def draw_random_batch(self,n_samples):
        random_x = draw_sobol_samples(
            bounds=self.bounds,n=1, q=n_samples
        ).squeeze(0)
        
        return random_x
        
    def selector(self,acquisition,q):
        new_x, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=self.bounds,
            q=q,
            num_restarts=64,
            raw_samples=1024, 
            sequential=False,
            equality_constraints = self.constraints
        )
        
        return new_x
        
    def initialize_model(self):
        self.model = SingleTaskGP(self.train_x, self.train_obj, 
            outcome_transform=Standardize(m=self.train_obj.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        return
        
        
    def suggest_next(self):
        logging.info('Getting suggestions for iteration %d'%self.iteration)
        
        if self.iteration==0:
            self.new_x = self.draw_random_batch(n_samples=self.batch_size)
        else:
            self.best_f = self.train_obj.max(axis=0).values
            self.acquisition = qExpectedImprovement(self.model, best_f = self.best_f)
            # optimize acquisition functions and get new observations
            self.new_x = self.selector(self.acquisition,q=self.batch_size)
        
        self.new_x_compspace = self.inp.inverse(self.new_x)
        logging.info('Iteration : %d, Samples to be made'%self.iteration)
        for i in range(self.batch_size):
            logging.info('%d\t%s'%(i, self.new_x_compspace[i,...].numpy()))
        
        logging.info('='*30)        
        
        self.iteration += 1
        
        return self.new_x_compspace
        
    def read_spectra(self,xlsx):
        xlsx = pd.read_excel(xlsx, index_col=0, engine='openpyxl')
        spectra = xlsx.values
        wavelengths = xlsx.index.to_numpy()
        
        return wavelengths, spectra.T

    def evaluate(self, xi, yi):
        """Scoring function at a given input location
        Uses the simulator sim to generate response spectra at a given locations
        and return a similarity score to target spectra
        """
        dist = self.metric(yi)

        self.expt[self.expt_id] = [self.iteration, xi, yi, dist]
    
        return torch.tensor([dist])
    
    def evaluate_batch(self, xb, yb):
        logging.debug('Current experiment id : %d'%self.expt_id)
        out = []
        for yi in zip(yb):
            out.append(self.evaluate(xb,yi))
            self.expt_id += 1
            
        return torch.stack(out, dim=0).to(**tkwargs)

    def update(self, xlsx, read_spectra=None):
        # re-initialize
        if read_spectra is None:
            self.wavelengths, self.spectra = self.read_spectra(xlsx)
        else:
            self.wavelengths, self.spectra = read_spectra(xlsx)
        self.new_obj = self.evaluate_batch(self.wavelengths, self.spectra)

        for i in range(self.batch_size):
            logging.debug('%d\t%s\t%s'%(i, self.new_x[i,...].numpy(), 
                                self.new_obj[i,...].numpy()))

        # update training points
        self.train_x = torch.cat([self.train_x, self.new_x])
        self.train_obj = torch.cat([self.train_obj, self.new_obj])
        self.initialize_model()
        fit_gpytorch_model(self.mll)

        return 
        
    def get_current_best(self):
        # obtain the current best from the model using posterior
        opt_x = self.selector(PosteriorMean(self.model), q=1)
        with torch.no_grad():
            posterior = self.model.posterior(torch.tensor(opt_x).to(**tkwargs))
            posterior_mean = posterior.mean.cpu().numpy()
        opt_x_compspace = self.inp.inverse(opt_x)
        self.best_loc.append(opt_x_compspace)
        logging.info('Iteration : %d Best estimate %s with predicted score : %s'%(self.iteration, 
            opt_x_compspace.numpy(), posterior_mean))
        
    def save(self):
        idir = self.savedir + '/%d'%(self.iteration-1)
        if not os.path.exists(idir):
            os.makedirs(idir)
        else:
            shutil.rmtree(idir)
            os.makedirs(idir)
            logging.debug('Iteriation %d has an existing directory in %s'%(self.iteration-1, idir))
            
        np.save(idir+'/new_x.npy',self.new_x_compspace.numpy())
        np.save(idir+'/new_obj.npy',self.new_obj.numpy())
        np.save(idir+'/train_x.npy',self.train_x.numpy())
        np.save(idir+'/train_obj.npy',self.train_obj.numpy())

        with open(idir + '/storage.pkl', 'wb') as handle:
            pickle.dump(self.expt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if hasattr(self, 'model'):
            np.save(idir+'/wavelengths.npy',self.wavelengths)
            np.save(idir+'/spectra.npy',self.spectra)
            torch.save(self.model.state_dict(), idir+'/model.pth')

        return 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        