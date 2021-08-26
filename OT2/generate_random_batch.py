import numpy as np
from datetime import datetime
import head 
from head.metrics import euclidean_dist
from configparser import ConfigParser

import torch
import os, sys 

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
iteration = int(config['BO']['iteration'])

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import ground_truth    

def generate_initial_data(n=6):
    points = torch.from_numpy(grid)
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    train_xid = torch.floor(soboleng.draw(n)*len(grid)).to(**tkwargs)
    train_x = points[train_xid.long(),:]
    
    return torch.squeeze(train_x).to(**tkwargs)
    
if __name__=='__main__':
    
    if not config['BO']['iteration']=='0':
        raise RuntimeError('This experiment has already been initialized...')

    problem = lambda s : batch_oracle(s)
    ref_point = torch.tensor([0,0]).to(**tkwargs)
    
    grid = np.loadtxt(savedir+'grid.txt', delimiter=',')    
    train_x = generate_initial_data(n=int(config['BO']['n_init_samples']))
    torch.save(train_x, savedir+'candidates_%d.pt'%iteration)
    np.savetxt(savedir+'candidates_%d.txt'%iteration, train_x.cpu().numpy())
    torch.save(train_x, savedir+'train_x.pt')
    print('Generated %d samples randomly'%int(config['BO']['n_init_samples']), 
        train_x.shape)
    print('Collect responses using OT2 and PlateReader...\nIn this case simply run or2_platereader.py')
    
    
    