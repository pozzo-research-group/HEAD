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
        
class L2:
    def __init__(self):
        """L2 space for functions defined on [0,1]-> R
        """ 
        
    def inner_product(self, y1, y2, x):
        """Compute inner product of two points in a L2 space
        """ 
        x_unit = (x-max(x))/(max(x)-min(x))
        
        return np.trapz(y1*y2, x=x_unit)
        
    def norm(self, y, x):
        """return norm of a function"""
        return np.sqrt(self.inner_product(y,y,x))
        
    def distance(self, v, w, x):
        """Compute distance between two tangent vectors"""
        
        return self.norm(v-w, x)
        
        
        
        
        
        
        
        
           