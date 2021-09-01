import os, sys
import numpy as np
import glob
import torch
import logging

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import ground_truth_moo

sys.path.append(os.path.join(os.path.dirname('./run_bo.py')))
from run_bo import initialize_model

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

import yaml

with open(os.path.abspath('./config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
savedir = config['Default']['savedir']

logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s%(message)s \t')

if __name__=='__main__':
        
    iteration = config['BO']['iteration']
    spectra_dir = savedir+'spectra_%d'%iteration
    logging.info('\tUpdating the models from %s'%spectra_dir)

    if iteration==0:
        train_obj = ground_truth_moo(spectra_dir)
        torch.save(train_obj, savedir+'train_obj.pt')
        logging.info('\tGround turth evaluations are: %s of shape %s'%(train_obj, repr(train_obj.shape)))
        logging.info('\tRandom batch evaluations are updated...')
    
    else:
        train_x = torch.load(savedir+'train_x.pt', map_location=tkwargs['device'])
        train_obj = torch.load(savedir+'train_obj.pt', map_location=tkwargs['device'])
        assert train_x.shape[0]==train_obj.shape[0], 'collected data shape does not match for iteration %d'%iteration
        
        # optimize acquisition functions and get new observations
        new_x = torch.load(savedir+'candidates_%d.pt'%iteration, map_location=tkwargs['device'])
        logging.info('\tEvaluting the responses in %s'%spectra_dir)
        new_obj = ground_truth_moo(spectra_dir)
        logging.info('\tGround turth evaluations are: %s'%(new_obj))

        # update training points
        logging.info('\tUpdating the training data from newly evaluated points')
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        assert train_x.shape[0]==train_obj.shape[0], 'Updated data shape does not match at iteration'%iteration
        best = train_obj.max(axis=0).values
        logging.info('\tBest %s'%(best))
        
        torch.save(train_x, savedir+'train_x.pt')
        torch.save(train_obj, savedir+'train_obj.pt')
        logging.info('\tNewly updated training data shape X : %s, Y : %s'%(train_x.shape, train_obj.shape))
    