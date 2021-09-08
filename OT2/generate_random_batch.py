import numpy as np
from datetime import datetime
import head 
from head.metrics import euclidean_dist
from configparser import ConfigParser

import torch
import os, sys, pdb

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import logger

sys.path.append(os.path.join(os.path.dirname('./stocks.py')))
from stocks import to_volume

logger = logger('generate_random_batch')

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
    logger.info('Converting concentrations to volumes to be despensed...')
    volume_df, seed_df = to_volume(train_x.numpy())
    volume_df.to_csv(savedir+'volume_%d.csv'%iteration)
    seed_df.to_csv(savedir+'seeds_%d.csv'%iteration)
    logger.info('Saved volumes and stocks to %s'%(savedir))
    
    logger.info('Initial random candidate points: %s'%(repr(train_x)))
    torch.save(train_x, savedir+'candidates_%d.pt'%iteration)
    np.savetxt(savedir+'candidates_%d.txt'%iteration, train_x.cpu().numpy())
    torch.save(train_x, savedir+'train_x.pt')
    logger.info('Generated %d samples randomly of shape %s'%(config['BO']['n_init_samples'],repr(train_x.shape)))
    logger.info('Collect responses using OT2 and PlateReader...')
    logger.info('Random sampling is successful')

    
    