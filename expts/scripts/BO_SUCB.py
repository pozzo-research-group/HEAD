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
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import LinearMCObjective, ScalarizedObjective
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import PosteriorMean

N_UVVIS_SAMPLES = 100
N_SAS_SAMPLES = 200
NUM_GRID_PERDIM = 100
BATCH_SIZE = 16
N_ITERATIONS = 10
NUM_RESTARTS = 20 
RAW_SAMPLES = 1024
N_INIT_SAMPLES = 8

R_mu = 20
R_sigma = 1e-2
SHAPE_PARAM = 0.67
SPECTRA = 'uvvis'

expt = {}
EXPT_ID = 0


T0 = time.time()
savedir = '../figures/SO_SUCB/'
if  os.path.exists(savedir):
	shutil.rmtree(savedir)
os.makedirs(savedir)


print('Setting up the targets ...')
sim = head.EmulatorMultiShape()
sim.make_structure(r_mu=R_mu,r_sigma=R_sigma)

fig, axs = plt.subplots(1,2,figsize=(4*2,4))

sim.plot_radii(axs[0])
axs[0].set_xlabel('radius')
axs[0].set_ylabel('Probability density')

if SPECTRA=='saxs':
    xt, yt = sim.get_saxs(shape_param = SHAPE_PARAM, n_samples=N_SAS_SAMPLES)
    axs[1].loglog(xt, yt)
    plt.setp(axs[1], xlabel='q (1/A)', ylabel='I(q)')
elif SPECTRA=='uvvis':
    xt, yt = sim.get_uvvis(shape_param = 0.67, n_samples=N_UVVIS_SAMPLES)
    axs[1].plot(xt, yt)
    plt.setp(axs[1], xlabel='wavelength (nm))', ylabel=r'I($\lambda$)')
    
fig.suptitle('r = '+','.join('%.2f'%i for i in sim.radii))
plt.savefig(savedir + '/targets.png', bbox_inches='tight')
plt.close()

print('Generated targets using the simulators...')

print('Defining search space using bounds...')
r_mu = [5,50]
r_sigma = [0,1]
shape_param = [0,1]
bounds = torch.tensor((r_mu, r_sigma, shape_param)).T.to(**tkwargs)
print('Bounds : ', bounds)

print('Defining oracles and their batch mode versions...')
def oracle(x):
    """Scoring function at a given input location
    Uses the simulator sim to generate response spectra at a given locations
    and return a similarity score to target spectra
    """
    global EXPT_ID
    x_np = x.cpu().numpy()    
    sim.make_structure(r_mu=x_np[0],
                       r_sigma=x_np[1])
    if SPECTRA=='saxs':
        xi, yi = sim.get_saxs(shape_param = x_np[2], 
                              n_samples=N_SAS_SAMPLES)
        dist = euclidean_dist(np.log10(yi),np.log10(yt))
    elif SPECTRA=='uvvis':
        xi, yi = sim.get_uvvis(shape_param = x_np[2], 
                               n_samples=N_UVVIS_SAMPLES)
        dist = euclidean_dist(yi,yt)
        
    expt[EXPT_ID] = [xi, yi, dist]
    EXPT_ID += 1
    
    return torch.tensor([dist])

def batch_oracle(x):
    out = []
    for xi in x.squeeze(1):
        out.append(oracle(xi))
    return torch.stack(out, dim=0).to(**tkwargs)
    
    
print('Generating random samples...')    

def draw_random_batch(n_samples=6):
    train_x = draw_sobol_samples(
        bounds=bounds,n=1, q=n_samples, 
        seed=torch.randint(2021, (1,)).item()
    ).squeeze(0)
    train_obj = batch_oracle(train_x)
    return train_x, train_obj


train_x, train_obj = draw_random_batch(n_samples=N_INIT_SAMPLES)
print('Generated %d samples randomly'%N_INIT_SAMPLES, train_x.shape, train_obj.shape)
for i in range(N_INIT_SAMPLES):
    print('%d\t%s\t%s'%(i, train_x[i,...].numpy(), train_obj[i,...].numpy()))  


print('Initializing the models and likelihood functions ...')
def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, 
    outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(train_x, train_obj)

# 3. Define acqusition function
obj = LinearMCObjective(weights=torch.tensor([1.0]).to(**tkwargs))

print('Defining a selector...')

def selector(f, q = BATCH_SIZE):
    new_x, _ = optimize_acqf(
        acq_function=f,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES, 
        sequential=False,
    )
    new_obj = batch_oracle(new_x)
    return new_x, new_obj

torch.manual_seed(2021)
print('Starting Bayesian Optimization ... ')
# run N_ITERATIONS rounds of BayesOpt after the initial random batch
print('Sampled ID \t Locations \t Objectives')
for iteration in range(1, N_ITERATIONS + 1):    
    print('Iteration : %d/%d'%(iteration, N_ITERATIONS))
    # fit the models
    fit_gpytorch_model(mll)

    # define the acquisition module
    acquisition = qUpperConfidenceBound(model, beta=0.1, objective=obj)
    
    # optimize acquisition functions and get new observations
    new_x, new_obj = selector(acquisition)
    for i in range(BATCH_SIZE):
        print('%d\t%s\t%s'%(i, new_x[i,...].numpy(), new_obj[i,...].numpy()))

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    # re-initialize
    mll, model = initialize_model(train_x, train_obj)

    best = train_obj.max(axis=0).values
    print('Best %.2f'%(best))


model.eval()
model.likelihood.eval()
print('Extracting best parameters from the surrogate model ...')
# obtain best sample and corresponding objective
objective = ScalarizedObjective(weights=torch.tensor([1.0]).to(**tkwargs))

print('Actual target : ', [R_mu, R_sigma])
opt_x, opt_obj = selector(PosteriorMean(model, objective=objective), q=1)
opt_x = opt_x.cpu().numpy().squeeze()
print('Optimal location: ',opt_x,
      '\nOptimal model scores: ', opt_obj.numpy())

print('Creating batchwise trace plots ...')
batch_number = torch.cat(
    [torch.zeros(N_INIT_SAMPLES), 
     torch.arange(1, N_ITERATIONS+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

for b in np.unique(batch_number):
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Batch number %d'%b)
    for i in np.argwhere(batch_number==b).squeeze():
        if SPECTRA=='uvvis':
            ax.plot(expt[i][0], expt[i][1], label='%.2f'%expt[i][2])
        elif SPECTRA=='saxs':
            ax.loglog(expt[i][0], expt[i][1], label='%.2f'%expt[i][2])
    ax.legend()
    plt.savefig(savedir + '/trace_b%d.png'%b, bbox_inches='tight')
    plt.close()

print('Time elapsed %.2f'%(time.time()-T0))