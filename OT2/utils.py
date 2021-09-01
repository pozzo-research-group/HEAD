import os, sys
import numpy as np
import glob
import torch
from head.metrics import euclidean_dist
import pdb

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }


import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
savedir = config['Default']['savedir']

import logging
logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s%(message)s \t')
  
def ground_truth_dls(dirname):
    files = sorted(glob.glob(dirname+'/*.txt'))
    out = []
    for file in files:
        if not 'dls' in file:
            continue
        st = np.loadtxt(savedir+'target_dls.txt', delimiter=',')
        si = np.loadtxt(dls, delimiter=',')
        dist = euclidean_dist(st, si) 
                
        out.append(dist)  
        
    return out 
    
def ground_truth_saxs(dirname):
    files = sorted(glob.glob(dirname+'/*.txt'))
    out = []
    for file in files:
        if not 'saxs' in file:
            continue
        st = np.loadtxt(savedir+'target_saxs.txt', delimiter=',')
        si = np.loadtxt(file, delimiter=',')
        dist = euclidean_dist(np.log10(si),np.log10(st)) 
        
        out.append(dist) 
                
    return out 
    
def ground_truth_uvvis(dirname):
    files = sorted(glob.glob(dirname+'/*.txt'))
    out = []
    for file in files:
        if not 'uvvis' in file:
            continue
        It = np.loadtxt(savedir+'target_uvvis.txt', delimiter=',')
        Ii = np.loadtxt(file, delimiter=',')
        dist = euclidean_dist(Ii,It)
                
        out.append(dist) 
                
    return out

def ground_truth_moo(dirname):
    if not os.path.exists(dirname):
        logging.error('\tSpectra directory does not exist...')
        raise RuntimeError
        
    scores = []
    
    if "dls" in config['BO']['objective']:
        scores.append(ground_truth_dls(dirname))
        
    if "saxs" in config['BO']['objective']:
        scores.append(ground_truth_saxs(dirname))

    if "uvvis" in config['BO']['objective']:
        scores.append(ground_truth_uvvis(dirname))
        
    return torch.from_numpy(np.asarray(scores).T).to(**tkwargs)

def get_best_sofar():
    from botorch.acquisition import PosteriorMean
    from botorch.acquisition.objective import ScalarizedObjective
    sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
    from run_bo import selector
    if len(config['BO']['objective'])==1:
        objective = None
    else:
        weights = config['BO']['weights']
        objective = ScalarizedObjective(weights=torch.tensor(weights).to(**tkwargs))
    
    train_obj = torch.load(savedir+'train_obj.pt')
    train_x = torch.load(savedir+'train_x.pt')

    sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
    from run_bo import load_models

    mll, model = load_models(train_x, train_obj)
    opt_x = selector(PosteriorMean(model, objective=objective), q=1)
    opt_x = opt_x.cpu().numpy().squeeze()

    print('Best sample from the optimization :', opt_x)

        
        