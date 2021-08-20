import numpy as np
from datetime import datetime
import head 
from head.metrics import euclidean_dist
from configparser import ConfigParser

import torch
import os 

tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

config = ConfigParser()
config.read("config.ini")
savedir = config['Default']['savedir']
    

def oracle(x):
    """Scoring function at a given input location
    Uses the simulator sim to generate response spectra at a given locations
    and return a similarity score to target spectra
    """
    
    x_np = x.cpu().numpy()
    sim = head.Emulator(n_structures=int(config['Modelling']['n_structures']))
    sim.make_structure(r_mu=x_np[0],r_sigma=x_np[1])
    q, si = sim.get_saxs(int(config['Modelling']['n_sas_samples']))
    wl, Ii = sim.get_spectrum(int(config['Modelling']['n_uvvis_samples']))
    
    st = np.loadtxt(savedir+'target_saxs.txt', delimiter=',')
    dist_sas = euclidean_dist(np.log10(si),np.log10(st))
    
    It = np.loadtxt(savedir+'target_uvvis.txt', delimiter=',')
    dist_uvvis = euclidean_dist(Ii,It)

    fname = "%s.npz"%config['Default']['oracle_call']
    np.savez(savedir+fname, 
        q=q, si=si, dist_sas=dist_sas, 
        dist_uvvis=dist_uvvis, wl=wl, Ii=Ii)
    
    config.set('BO', 'oracle_call', str(int(config['BO']['oracle_call'])+1))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    
    
    return torch.from_numpy(np.asarray([dist_sas, dist_uvvis]))

def batch_oracle(x):
    out = []
    for xi in x.squeeze(1):
        out.append(oracle(xi))
    return torch.stack(out, dim=0).to(**tkwargs)

def generate_initial_data(n=6):
    points = torch.from_numpy(grid)
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    train_xid = torch.floor(soboleng.draw(n)*len(grid)).to(**tkwargs)
    train_x = points[train_xid.long(),:]
    train_obj = problem(train_x)
    
    return torch.squeeze(train_x).to(**tkwargs), torch.squeeze(train_obj).to(**tkwargs)
    
if __name__=='__main__':
    
    if not config['BO']['iteration']=='0':
        raise RuntimeError('This experiment has already been initialized...')

    problem = lambda s : batch_oracle(s)
    ref_point = torch.tensor([0,0]).to(**tkwargs)
    
    grid = np.loadtxt(savedir+'grid.txt', delimiter=',')    
    train_x, train_obj = generate_initial_data(n=int(config['BO']['n_init_samples']))

    torch.save(train_x, savedir+'train_x.pt')
    torch.save(train_x, savedir+'train_obj.pt')
    print('Generated %d samples randomly'%int(config['BO']['n_init_samples']), 
        train_x.shape, train_obj.shape)
    
    
    