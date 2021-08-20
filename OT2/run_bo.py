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
sys.path.append(os.path.join(os.path.dirname('./generated_random_batch.py')))
from generate_random_batch import batch_oracle

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
    
# 2. define your model
def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
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


# 4. define a optimization routine for acqusition function 

def selector(f, q):
    grid = np.loadtxt(savedir+'grid.txt', delimiter=',')  
    choices = torch.from_numpy(grid).to(**tkwargs)
    new_x, _ = optimize_acqf_discrete(
        acq_function = f,
        q=q,
        choices = choices
    )
    new_obj = problem(new_x)
    return new_x, new_obj


# 5. define the opitmization loop
if __name__=='__main__':
    
    config.set('BO', 'iteration', str(int(config['BO']['iteration'])+1))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
        
    iteration = int(config['BO']['iteration'])
    
    problem = lambda s : batch_oracle(s)
    ref_point = torch.tensor([0,0]).to(**tkwargs)
    
    # load the train data collected so far
    train_x = torch.load(savedir+'train_x.pt', map_location=torch.device('cpu'))
    train_obj = torch.load(savedir+'train_obj.pt', map_location=torch.device('cpu'))
    
    mll, model = initialize_model(train_x, train_obj)
    torch.save(model.state_dict(), savedir+'model_state.pth')
    torch.save(mll.state_dict(), savedir+'mll_state.pth')
        
    print('Iteration : %d/%d'%(iteration, int(config['BO']['n_iterations'])))

    mll, model = load_models(train_x, train_obj)
    
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition modules using a QMC sampler
    acquisition = acq_fun(model)

    # optimize acquisition functions and get new observations
    new_x, new_obj = selector(acquisition, q= int(config['BO']['batch_size']))

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    best = train_obj.max(axis=0).values
    print('Best SAS distance : %.2f, Best UVVis distance : %.2f'%(best[0], best[1]))
    
    torch.save(train_x, savedir+'train_x.pt')
    torch.save(train_x, savedir+'train_obj.pt')
    torch.save(model.state_dict(), savedir+'model_state.pth')
    torch.save(mll.state_dict(), savedir+'mll_state.pth')
    

