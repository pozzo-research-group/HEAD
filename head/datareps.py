import numpy as np
from skfda.representation.grid import FDataGrid
import pdb

class Spectra1D:
    """One-dimensional spectral data class
    This class is the base class for spectral data represented as discrete evaluations over a one-dimenional grid
    Examples include, UV-vis, X-ray scattering etc
    """
    
    def __init__(self, grid, grid_evals,*args,**kwargs):
        self.fd = FDataGrid(grid_evals, grid, *args,**kwargs)
        
    def plot(self,*args,**kwargs):
        self.fig = self.fd.plot(*args,**kwargs)
        self.ax = self.fig.gca()
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.ax.yaxis.set_ticks_position('left')
        self.ax.xaxis.set_ticks_position('bottom')
        
        
class UVVis(Spectra1D):
    def __init__(self, wavelengths, absorptions,
                 dataset_name ='UV-vis',*args,**kwargs):
        super().__init__(wavelengths, absorptions, *args,**kwargs)
    
    def decorate(self):
        self.ax.set_xlabel('wavelength')
        self.ax.set_ylabel('Intensity')
                      
        