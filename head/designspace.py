import numpy as np
import pdb
from sklearn.neighbors import kneighbors_graph

class Euclidean:
    def __init__(self, *x):
        """Euclidean space in d dimensions (```\mathbb{R}^d```)
        
        X = np.linspace(1,2, num=10) 
        Y = np.linspace(1,2, num=10)
        grid = head.Euclidean(X,Y)
        
        """
        self.mesh = np.meshgrid(*x)
        self.space = np.vstack(map(np.ravel, self.mesh)).T
        self.d = len(x)
        
        if not self.d==self.space.shape[1]:
            raise ValueError('The dimension of the space should be {} not {}'.format(self.d, 
                                                                                     self.space.shape[1]))
    def __getitem__(self,i):
        
        return self.space[i].squeeze()
    
    def __len__(self):
        return len(self.space)

    def sample(self, n_samples=1, method='random'):
        """Generate a random sample from the space
        """
        idx = np.random.random_integers(int(len(self.space)-1), size=n_samples)
        
        return self.space[idx,:].squeeze()
    
    def shape(self):
        return self.space.shape
    
    def to_graph(self,k=8):
        A = kneighbors_graph(self.space, k, 
                             mode='connectivity', include_self=True)
        return A
    
    
class Hyperplane(Euclidean):
    """Hyperplan that is obtained by constraining the components to sum upto 1
    Used for volume and wieght fractions.
    
    x = np.linspace(0,1, num=10) 
    y = np.linspace(0,1, num=10) 
    hplane = head.Hyperplane(x,y)
    
    """
    def __init__(self,*x):
        super().__init__(*x)
        inplane = [self.belongs(p) for p in self.space]
        self.space = self.space[inplane,:]
        
    def belongs(self, p):
        return np.isclose(np.sum(p),1.0, atol=1e-3)
    
        