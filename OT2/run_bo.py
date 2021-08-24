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

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
iteration = int(config['BO']['iteration'])
    
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
obj = LinearMCObjective(weights=torch.tensor([0.5, 0.5]).to(**tkwargs))
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
  
    return new_x


# 5. define the opitmization loop
if __name__=='__main__':
    # define a ground truth function 
    config.set('BO', 'iteration', str(int(config['BO']['iteration'])+1))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
        
    iteration = int(config['BO']['iteration'])
    n_iterations = int(config['BO']['n_iterations'])
    
    if iteration>n_iterations:
        raise RuntimeError('Maximum number of iterations reached')
    
    print('Iteration : %d/%d'%(iteration, int(config['BO']['n_iterations'])))

    # load the train data collected so far
    train_x = torch.load(savedir+'train_x.pt', map_location=tkwargs['device'])
    train_obj = torch.load(savedir+'train_obj.pt', map_location=tkwargs['device'])
    mll, model = initialize_model(train_x, train_obj)  
     

    mll, model = load_models(train_x, train_obj)
    
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition modules
    acquisition = acq_fun(model)

    # optimize acquisition functions and get new observations
    new_x = selector(acquisition, q= int(config['BO']['batch_size']))
    torch.save(new_x, savedir+'candidates_%d.pt'%iteration)
    



    

