import numpy as np
import pdb

class Euclidean:
    def __init__(self, *x):
        """Euclidean space in d dimensions (```\mathbb{R}^d```)
        """
        self.mesh = np.meshgrid(*x)
        self.space = np.vstack(map(np.ravel, self.mesh)).T
        self.d = len(x)
        
        if not self.d==self.space.shape[1]:
            raise ValueError('The dimension of the space should be {} not {}'.format(self.d, 
                                                                                     self.space.shape[1]))
    def __getitem__(self,i):
        
        return self.space[i].squeeze()
    
    def sample(self, n_samples=1, method='random'):
        """Generate a random sample from the space
        """
        idx = np.random.random_integers(int(len(self.space)-1), size=n_samples)
        
        return self.space[idx,:].squeeze()