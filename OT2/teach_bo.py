import os, sys
import numpy as np
import glob
from configparser import ConfigParser
import torch

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import ground_truth_moo

sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
from run_bo import initialize_model

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']

if __name__=='__main__':
        
    iteration = int(config['BO']['iteration'])
    spectra_dir = savedir+'spectra_%d'%iteration
    print('Updating the models from %s'%spectra_dir)
    
    train_x = torch.load(savedir+'train_x.pt', map_location=tkwargs['device'])
    train_obj = torch.load(savedir+'train_obj.pt', map_location=tkwargs['device'])
    
    # optimize acquisition functions and get new observations
    new_x = torch.load(savedir+'candidates_%d.pt'%iteration, map_location=tkwargs['device'])
    new_obj = ground_truth_moo(spectra_dir)

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    best = train_obj.max(axis=0).values
    print('Best SAS distance : %.2f, Best UVVis distance : %.2f'%(best[0], best[1]))
    
    torch.save(train_x, savedir+'train_x.pt')
    torch.save(train_obj, savedir+'train_obj.pt')
    