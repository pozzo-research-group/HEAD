import numpy as np
import matplotlib.pyplot as plt

from pyGDM2 import (core, propagators, fields, 
                    materials, linear, structures, 
                    tools, visu)

from scipy import stats
import pdb

import warnings
warnings.filterwarnings("ignore")

from sasmodels.data import empty_data1D, plot_theory
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

class Emulator:
    def __init__(self, use_mean = True, n_structures=None):
        
        # randmoly select number of structures
        if n_structures is None:
            self.n_structures = stats.randint.rvs(2,6,size=1)[0]
        else:
            self.n_structures = n_structures
            
        rng = np.random.default_rng()
        self.XY_ = rng.standard_normal(size=(self.n_structures, 2))

        if use_mean:
            self.make_polydisperse = self._average_profiles
        else:
            self.make_polydisperse = self._make_polydisperse
        
        
    def make_structure(self, r_mu, r_sigma, spatial= False):
        """Create a structure with spatially distributed nano-sphere by 
        sampling radius from a normal distribution
        Inputs:
            r_mu, r_sigma : mean and variance of the normal distribution
            
        NOTE : pyGDM2 define radius as step*number; r_mu and r_sigma corresponds to number;
        step is set to be 15 by default
        
        """
        
        # sample radius from a log-normal distribution
        # pyGDM2 defines radius as step*number of particles thus self.radii is a number
        # while the actual radii of the sphere is self.step*self.radii
        #print('Structure parameters : ', r_mu, r_sigma, self.n_structures)
        self.step = 15
        self.r_mu = r_mu
        self.r_sigma = r_sigma
        self.spatial = spatial
        self.radii_dist_kwargs = {'s': self.r_sigma, 'loc':self.r_mu, 'scale':1}
        self.radii_dist = stats.lognorm(**self.radii_dist_kwargs) 
        self.radii = stats.lognorm.rvs(size=self.n_structures, **self.radii_dist_kwargs)
        self.radii_nrs = self.radii/self.step
   
    def _make_pydgm2_geom(self):
        # define a scale to use for distributing particles spatially
        spatial_scale = (self.r_mu+self.r_sigma)*self.step*5

        # unsure whether to vary this or to make this a user defined variable, in which case we are
        # massively increasing the design space 
        # For now, the impact of the spatial distribution is not considered
        # This assumption is valid for SAS profiles where the spatial variation is not accounted for
        # pyGDM2 however, take this into account to return a spectrum
        self.XY= spatial_scale*self.XY_
        self.material = materials.gold()
        geom_list = []

        for i,(x,y) in enumerate(self.XY):
            _geo = structures.sphere(self.step, R=self.radii_nrs[i], mesh='hex')
            if self.spatial:
                _geo = structures.shift(_geo, [x, y, 0])
            geom_list.append(_geo)
            
        if self.spatial:
            self.geometry = structures.combine_geometries(geom_list)
        else:
            self.geometry = geom_list
        
        return self    
        
    def plot_structure2d(self, ax=None):
        if not self.spatial:
            return
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.scatter(self.geometry[:,0], self.geometry[:,1])
        ax.scatter(self.XY[:,0], self.XY[:,1], marker='x')
        ax.axis('equal')
        
        return
    
    def plot_radii(self, ax = None):
        if ax is None:
           fig, ax = plt.subplots()
        x_min =  stats.lognorm.ppf(0.01, **self.radii_dist_kwargs)
        x_max =  stats.lognorm.ppf(0.99, **self.radii_dist_kwargs)
        x = np.linspace(x_min, x_max, 1000)
        
        ax.plot(x, self.radii_dist.pdf(x), 'k-', lw=2)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        return ax
    
    def _simulate_uvvis(self, struct, wavelengths):
        self._make_pydgm2_geom()
        field_generator = fields.plane_wave
        kwargs = dict(theta=0, inc_angle=180)

        efield = fields.efield(field_generator,
                       wavelengths=wavelengths, kwargs=kwargs)
        n1 = n2 = 1.0
        dyads = propagators.DyadsQuasistatic123(n1=n1, n2=n2)

        sim = core.simulation(struct, efield, dyads)
        sim.scatter(verbose=False)
        field_kwargs = tools.get_possible_field_params_spectra(sim)

        config_idx = 0
        wl, spectrum = tools.calculate_spectrum(sim,
                            field_kwargs[config_idx], linear.extinct)
        
        abs_ = spectrum.T[2]/np.max(spectrum.T[2])
        
        return abs_
            
    def _get_Iq(self, q, radius):
        """Obtain a SAS profile using sasview models
        
        Reproduced from : 
        https://github.com/SasView/sasmodels/blob/462a07ed6413ea731e6a4f9f0a5edb2c42dcda00/sasmodels/models/_spherepy.py#L82
        
        """
        V = (4/3)*np.pi*(radius**3)
        del_rho = 5
        qr = q * radius
        sn, cn = np.sin(qr), np.cos(qr)
        bes = 3 * (sn - qr * cn) / qr ** 3 
        bes[qr == 0] = 1
        fq = bes * del_rho * V
        Iq = 1.0e-4 * fq ** 2
        
        return Iq
        
    def _make_polydisperse(self, F, x):
        pdf = self.radii_dist.pdf(self.radii)
        integrand = np.asarray([F[i]*pdf[i] for i in range(len(self.radii_nrs))])
        indx = np.argsort(self.radii)
        integral = np.asarray([np.trapz(integrand[indx,i], x = self.radii[indx]) for i in range(len(x))])
        
        # obtain the scale parameter
        # It is not possible to assess the scale value uniformly across SAS and UVVis modelling
        # so this is set to one for simplicity
        scale = 1.0
        
        return scale*integral  
    
    def _average_profiles(self, F, x):
        
        return np.mean(np.asarray(F), axis=0)
    
               
    def get_spectrum(self, n_samples=10):
        """ Obtain a simulated absorption spectra for a hexagonal nanorod mesh
        """
        wl = np.linspace(400, 1000, n_samples)
        if self.spatial:
            struct = structures.struct(self.step, self.geometry, 
                self.material, verbose=False)
            
            abs_ = self._simulate_uvvis(struct,wavelengths)
            
            return wl, abs_
        else:
            abs_all = []
            for geom in self.geometry:
                struct = structures.struct(self.step, geom, 
                self.material, verbose=False)
                abs_ = self._simulate_uvvis(struct, wl)
                abs_all.append(abs_)
                
            abs_pd = self.make_polydisperse(abs_all, wl)
                
            return wl, abs_pd

    def get_saxs(self, n_samples=200):
        """Obtain a SAS profile with polydispersity
        
        Implemented following the description provided in:
        https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html
        """
        q = np.logspace(np.log10(1e-3), np.log10(1), n_samples)
        Iqs = [self._get_Iq(q, ri) for ri in self.radii]
        Iq_pd = self.make_polydisperse(Iqs, q)
        
        return q, Iq_pd
        
    
class EmulatorMultiShape(Emulator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)    

    def _get_shape(self, x):
        if x<0.5:
            return "sphere"
        elif x>=0.5:
            return "cylinder"
        else:
            raise RuntimeError('Value %.2f is not a valid shape parameter'%x)
            
    def _sasmodels(self, q, shape_param, radius):
        data = empty_data1D(q, resolution=0.05)
        kernel = load_model(self._get_shape(shape_param))
        f = DirectModel(data, kernel)

        return f(radius=radius)
    
    def _pygdm2(self, shape_param):

        self.material = materials.gold()
        geom_list = []
        
        if self._get_shape(shape_param)=='sphere':
            geom_base = lambda r : structures.sphere(self.step, r, mesh='hex')
            
        elif self._get_shape(shape_param)=='cylinder':
            geom_base = lambda r : structures.nanorod(self.step, r*10, r, mesh='hex')
            
        for ri in self.radii_nrs:
            geom_list.append(geom_base(ri))
            
        self.geometry = geom_list
        
        return self
        
    def get_uvvis(self, shape_param, n_samples=200):
        self._pygdm2(shape_param)
        wl = np.linspace(400, 1000, n_samples)
        abs_all = []
        for geom in self.geometry:
            struct = structures.struct(self.step, geom, 
            self.material, verbose=False)
            abs_ = self._simulate_uvvis(struct, wl)
            abs_all.append(abs_)
            
        abs_pd = self.make_polydisperse(abs_all, wl)
            
        return wl, abs_pd           
        
    def get_saxs(self, shape_param, n_samples=200):
        """Obtain a SAS profile with polydispersity
        
        Implemented following the description provided in:
        https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html
        """
        q = np.logspace(np.log10(1e-3), np.log10(1), n_samples)
        Iqs = [self._sasmodels(q, shape_param, ri) for ri in self.radii]
        Iq_pd = self.make_polydisperse(Iqs, q)
        
        return q, Iq_pd

        
        
        
        
        
        
        
        
        
        