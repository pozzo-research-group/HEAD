import os, sys
import numpy as np
import glob
import torch
from configparser import ConfigParser
from head.metrics import euclidean_dist
import pdb

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']


def ground_truth(dirname):
    files = sorted(glob.glob(dirname+'/*.txt'))
    n_files = len(files)//2
    out = []
    for i in range(n_files):
        st = np.loadtxt(savedir+'target_saxs.txt', delimiter=',')
        si = np.loadtxt(dirname+'/%d_saxs.txt'%i, delimiter=',')
        dist_sas = euclidean_dist(np.log10(si),np.log10(st)) 
        
        It = np.loadtxt(savedir+'target_uvvis.txt', delimiter=',')
        Ii = np.loadtxt(dirname+'/%d_uvvis.txt'%i, delimiter=',')
        dist_uvvis = euclidean_dist(Ii,It)
                
        scores = torch.from_numpy(np.asarray([dist_sas, dist_uvvis]))
        out.append(scores) 
                
    return torch.stack(out, dim=0).to(**tkwargs) 