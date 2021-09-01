import numpy as np
from datetime import datetime
import head 
from head.metrics import euclidean_dist
from configparser import ConfigParser

import torch
import os, sys 

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

import logging
logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s%(message)s \t')

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

savedir = config['Default']['savedir']
iteration = config['BO']['iteration']
   

def generate_initial_data(n=6):
    points = torch.from_numpy(grid)
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    train_xid = torch.floor(soboleng.draw(n)*len(grid)).to(**tkwargs)
    train_x = points[train_xid.long(),:]
    
    return train_x.squeeze().to(**tkwargs)
    
if __name__=='__main__':
    
    if not config['BO']['iteration']==0:
        raise RuntimeError('This experiment has already been initialized...')
        
    if os.path.exists(savedir+'/candidates_0.pt'):
        raise RuntimeError('This experiment has already been initialized...')
        
    problem = lambda s : batch_oracle(s)
    ref_point = torch.tensor([0,0]).to(**tkwargs)
    
    grid = np.loadtxt(savedir+'grid.txt', delimiter=',')    
    train_x = generate_initial_data(n=config['BO']['n_init_samples'])
    logging.info('\tInitial random candidate points: %s'%(repr(train_x)))
    torch.save(train_x, savedir+'candidates_%d.pt'%iteration)
    np.savetxt(savedir+'candidates_%d.txt'%iteration, train_x.cpu().numpy())
    torch.save(train_x, savedir+'train_x.pt')
    logging.info('\tGenerated %d samples randomly of shape %s'%(config['BO']['n_init_samples'],repr(train_x.shape)))
    logging.info('\tCollect responses using OT2 and PlateReader...')
    logging.info('\tRandom sampling is successful')

    
    