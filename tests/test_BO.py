import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from head import opentrons
import pandas as pd
import numpy as np
import shutil
import unittest

class Simulator:
    def __init__(self):
        self.domain = np.linspace(-5,5,num=100)
        
    def generate(self, mu, sig):
        scale = 1/(np.sqrt(2*np.pi)*sig)
        return scale*np.exp(-np.power(self.domain - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def process_batch(self, Cb, fname):
        out = []
        for c in Cb:
            out.append(self.generate(*c))
        out = np.asarray(out)
        df = pd.DataFrame(out.T, index=self.domain)
        df.to_excel(fname, engine='openpyxl')
        
        return 
    
    def make_target(self, ct):
        return self.domain, self.generate(*ct)


sim = Simulator()
target = np.array([-2,0.5])
xt, yt = sim.make_target(target)


Cmu = [-5,5]
Csig = [0.1,3.5]
bounds = [Cmu, Csig]

optim = opentrons.Optimizer(xt, yt, bounds, 
                            savedir = '../data2/', batch_size=4)


optim.save()
C0 = np.load('../data2/0/new_x.npy')
sim.process_batch(C0, '../data2/0.xlsx')
optim.update('../data2/0.xlsx')
optim.save()


for i in range(1,2):
    # iteration i selection
    optim.suggest_next()
    optim.save()
    # simulate iteration i new_x 
    Ci = np.load('../data2/%d/new_x.npy'%i)
    sim.process_batch(Ci, '../data2/%d.xlsx'%i)
    optim.update('../data2/%d.xlsx'%i)
    optim.save()


train_obj = np.load('../data2/1/train_obj.npy')
assert train_obj.shape == (8,1), 'Train_obj does not have the correct size'
assert train_obj.any() > 0, 'Train_obj has positive numbers' 


new_x = np.load('../data2/1/new_x.npy')
assert new_x.shape == (4,2), 'new_x does not have the correct size'
assert new_x[:,0].any() < Cmu[1] or new_x[:,0].any() > Cmu[0], 'guesses are out of bounds'
assert new_x[:,1].any() < Csig[1] or new_x[:,1].any() > Csig[0], 'guesses are out of bounds'


shutil.rmtree('../data2')

