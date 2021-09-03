import torch
torch.manual_seed(0)
from configparser import ConfigParser
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import LinearMCObjective

import os, sys
import numpy as np
import logging
import pdb

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

savedir = config['Default']['savedir']
iteration = config['BO']['iteration']

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import logger
logger = logger('read_saxs')
        
# 2. define your model
def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    torch.save(model.state_dict(), savedir+'model_state.pth')
    torch.save(mll.state_dict(), savedir+'mll_state.pth')
    
    return mll, model

def load_models(train_x, train_obj):
    state_dict = torch.load(savedir+'model_state.pth')
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    model.load_state_dict(state_dict)
    
    state_dict = torch.load(savedir+'mll_state.pth')
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.load_state_dict(state_dict)
    
    return mll, model

# 3. Define acqusition function
if len(config['BO']['objective'])==1:
    obj = None
else:
    weights = config['BO']['weights']
    obj = LinearMCObjective(weights=torch.tensor(weights).to(**tkwargs))
    
acq_fun = lambda model: qUpperConfidenceBound(model, beta=0.1, objective=obj)


# 4. define a optimization routine for acqusition function i.e. a selector

def selector(f,q):
    grid = np.loadtxt(savedir+'grid.txt', delimiter=',')  
    choices = torch.from_numpy(grid).to(**tkwargs)
    new_x, _ = optimize_acqf_discrete(
        acq_function = f,
        q=q,
        choices = choices
    )
    torch.save(new_x, savedir+'candidates_%d.pt'%iteration)
    np.savetxt(savedir+'candidates_%d.txt'%iteration, new_x.cpu().numpy())
    
    return new_x


# 5. define the opitmization loop
if __name__=='__main__':
    config['BO']['iteration'] = config['BO']['iteration']+1
    with open(os.path.abspath('./config.yaml'), 'w') as fp:
        yaml.dump(config, fp)
        
    iteration = config['BO']['iteration']
    n_iterations = config['BO']['n_iterations']
    
    if iteration>n_iterations:
        sys.path.append(os.path.join(os.path.dirname('./utils.py')))
        from utils import get_best_sofar
        get_best_sofar()
        logger.info('Maximum number of iterations reached')
    
    logger.info('Iteration : %d/%d'%(iteration, config['BO']['n_iterations']))

    # load the train data collected so far
    train_x = torch.load(savedir+'train_x.pt', map_location=tkwargs['device'])
    train_obj = torch.load(savedir+'train_obj.pt', map_location=tkwargs['device'])
    
    logger.info('initializing the GP surrogate model using %d samples'%(train_x.shape[0]))
    mll, model = initialize_model(train_x, train_obj)  
     
    logger.info('loading the GP surrogate model')
    mll, model = load_models(train_x, train_obj)
    
    # fit the models
    logger.info('Fitting the GP surrogate model hyper-parameters')
    fit_gpytorch_model(mll)

    # define the acquisition modules
    acquisition = acq_fun(model)

    # optimize acquisition functions and get new observations
    logger.info('Selecting the next best samples to query')
    new_x = selector(acquisition, q= int(config['BO']['batch_size']))
    logger.info('Newly selected points are %s \nof shape %s'%(new_x, new_x.shape[0]))
    torch.save(new_x, savedir+'candidates_%d.pt'%iteration)
    



    

