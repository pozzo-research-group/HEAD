import numpy as np
import pdb
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay


class Euclidean:
    def __init__(self, points):
        """Euclidean space in d dimensions (```\mathbb{R}^d```)
        
        points = np.random.rand(10,3)
        Rd = Euclidean(points)
        
        """
        self.points = points
        
    def __getitem__(self,i):
        return self.points[i].squeeze()
    
    def __len__(self):
        return len(self.points)

    def sample(self, n_samples=1, method='random'):
        """Generate a random sample from the space
        """
        idx = np.random.random_integers(int(len(self)-1), size=n_samples)
        
        return self.points[idx,:].squeeze()
    
    @property
    def shape(self):
        return self.points.shape
    
    def to_graph(self,k=8):
        A = kneighbors_graph(self.points, k, 
                             mode='connectivity', include_self=True)
        return A
        
    def __iter__(self):
        self.id = 0
        return self 
        
    def __next__(self):
        if self.id<len(self):
            out = self.__getitem__(self.id)
            self.id += 1
        else:
            raise StopIteration
        
        return out

class Grid(Euclidean):
    def __init__(self, *x):
        """Euclidean space in d dimensions (```\mathbb{R}^d```)
        
        X = np.linspace(1,2, num=10) 
        Y = np.linspace(1,2, num=10)
        grid = Grid(X,Y)
        
        """
        self.mesh = np.meshgrid(*x)
        points = np.vstack(map(np.ravel, self.mesh)).T
        super().__init__(points)

    
    
class Hyperplane(Euclidean):
    """Hyperplan that is obtained by constraining the components to sum upto 1
    Used for volume and wieght fractions.
    
    x = np.linspace(0,1, num=10) 
    y = np.linspace(0,1, num=10) 
    hplane = Hyperplane(x,y)
    
    """
    def __init__(self,*x):
        super().__init__(*x)
        inplane = [self.belongs(p) for p in self.points]
        self.points = self.points[inplane,:]
        
    def belongs(self, p):
        return np.isclose(np.sum(p),1.0, atol=1e-3)
        
class ConvexHull(Euclidean):
    """Convex Hull of set of Euclidean points
    
    Inputs:
    -------
        points : set of points for which a convex hull needs to be constructed
        N_finegrid : discretization (interms of numer of points per dimension) 
                     along each dimension to sample more points inside convex hull
    
     Example:
        rng = np.random.default_rng()
        points = rng.random((30, 2))
        space = ConvexHull(points,N_finegrid=25)

        plt.plot(points[:,0], points[:,1], 'ro')
        for simplex in space.hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.plot(space.space[:,0],space.space[:,1],'bo')
        
    """
    def __init__(self,points, N_finegrid=10):
        self.Nfgd = N_finegrid
        self.input_points = points
        self.hull = CH(self.input_points)
        self._delaunay = Delaunay(self.input_points)
        fgd = self._get_euclidean_boundaries()
        in_hull = [self.inhull(p) for p in fgd.points]
        hull_points = fgd.points[in_hull,:]
        super().__init__(hull_points)
        
    def _get_euclidean_boundaries(self):
        x = []
        for pi in self.input_points.T:
            min_ = pi.min()
            max_ = pi.max()       
            x.append(np.linspace(min_,max_, num=self.Nfgd))
        
        return Grid(*x)
        
    def inhull(self,point):
        return self._delaunay.find_simplex(point)>=0
        
        
        