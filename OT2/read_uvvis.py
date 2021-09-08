import numpy as np
import head 
import os, sys, pdb
import torch
import logging
import yaml
import pandas as pd 
import argparse

sys.path.append(os.path.join(os.path.dirname('./utils.py')))
from utils import logger
logger = logger('read_saxs')

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }


with open(os.path.abspath('./config.yaml'), 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	
savedir = config['Default']['savedir']
iteration = config['BO']['iteration']

logger.info('\tGenerating spectra from iteration %d'%iteration)
spectra_dir = savedir+'spectra_%d'%iteration

if os.path.exists(spectra_dir):
	logger.info('It appears that spectra for %d iteration has already been collected in %s'%(iteration, spectra_dir))
else:
	os.makedirs(spectra_dir)

parser = argparse.ArgumentParser(description='Process some user inputs for teach_bo')
parser.add_argument('--xlsx', metavar='xlsx', type=str,
                    help='a directory with the collected spectras to be updated into the model',
                    default = None)
args = parser.parse_args()

if args.xlsx is None:
	raise RuntimeError('You should input a .xlsx file with the spectra')

spectra = pd.read_excel(args.xlsx, index_col=0, engine='openpyxl')  
for i,(_,Ii) in enumerate(spectra.items()):
	fname = spectra_dir+'/%d_uvvis.txt'%i
	logger.info('reading and saving UV-Vis spectra in column %d from %s into %s'%(i, args.xlsx, fname))
	np.savetxt(fname, Ii, delimiter=',')
    