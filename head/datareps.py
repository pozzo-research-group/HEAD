import numpy as np
import pdb
from scipy.spatial import distance
import matplotlib.pyplot as plt
            
class SymmetricMatrices:
    def __init__(self, x, y, num_filters=4):
        self.num_filters = num_filters
        self.ind_split = np.split(np.arange(len(x)), num_filters)
        self.x_splits = np.split(x, self.num_filters)
        self.y_splits = self._get_splits(y)
        self.Id = self.get_rep(y)
        
    def plot_filters(self):
        fig, axs = plt.subplots(1, self.num_filters, 
                                figsize=(2*self.num_filters, 2))
        fig.subplots_adjust(wspace=0.4)
        for i, (xs,ys) in enumerate(zip(self.x_splits, self.y_splits)):
            axs[i].plot(xs, ys)
        return fig, axs
    
    def _get_splits(self, s):
        s_splits = []
        for ind in self.ind_split:
            s_splits.append(s[ind])
        return np.vstack(s_splits)
    
    def get_rep(self, s):
        s_splits = self._get_splits(s)
        d = distance.pdist(s_splits, distance.correlation)
        return distance.squareform(d)
    
    def distance(self, s):
        M_s = self.get_rep(s)
        return np.linalg.norm(self.Id-M_s,ord='fro')
        

        
        
        
        
        
           