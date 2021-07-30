import numpy as np
import matplotlib.pyplot as plt

from pyGDM2 import (core, propagators, fields, 
                    materials, linear, structures, 
                    tools, visu)

from scipy import stats
import pdb

import warnings
warnings.filterwarnings("ignore")

class Emulator:
    def __init__(self):
        
        # randmoly select number of structures
        self.n_structures = stats.randint.rvs(2,6,size=1)[0]
        rng = np.random.default_rng()
        self.XY_ = rng.standard_normal(size=(self.n_structures, 2))
        
    def make_structure(self, r_mu, r_sigma):
        """Create a structure with spatially distributed nano-sphere by 
        sampling radius from a normal distribution
        Inputs:
            r_mu, r_sigma : mean and variance of the normal distribution
            
        NOTE : pyGDM2 define radius as step*number; r_mu and r_sigma corresponds to number
        step is set to be 20 by default
        
        """
        
        # sample radius from a Gaussian distribution
        # pyGDM2 defines radius as step*number of particles thus self.radii is a number
        # while the actual radii of the sphere is self.step*self.radii
        self.step = 15
        self.r_mu = r_mu
        self.r_sigma = r_sigma
        self.radii_dist = stats.lognorm(s = self.r_sigma, loc=self.r_mu,scale=1) 
        self.radii = stats.lognorm.rvs(s = self.r_sigma, loc=self.r_mu,
            scale=1,size=self.n_structures)

        # define a scale to use for distributing particles spatially
        spatial_scale = (self.r_mu+self.r_sigma)*self.step*5

        # unsure whether to vary this or to make this a user defined variable, in which case we are
        # massively increasing the design space 
        # For now, the impact of the spatial distribution is not considered
        # This assumption is valid for SAS profiles where the spatial variation is not accounted for
        # pyGDM2 however, take this into account to return a spectrum
        
        self.XY= spatial_scale*self.XY_

        geom_list = []

        for i,(x,y) in enumerate(self.XY):
            _geo = structures.sphere(self.step, R=self.radii[i], mesh='hex')
            _geo = structures.shift(_geo, [x, y, 0])
            geom_list.append(_geo)

        self.geometry = structures.combine_geometries(geom_list)
        self.material = materials.gold()
        self.struct = structures.struct(self.step, self.geometry, self.material, verbose=False)
        
        return self 
        
    def plot_structure2d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.scatter(self.geometry[:,0], self.geometry[:,1])
        ax.scatter(self.XY[:,0], self.XY[:,1], marker='x')
        ax.axis('equal')
        
        return
         
    def get_spectrum(self):
        """ Obtain a simulated absorption spectra for a hexagonal nanorod mesh
        """

        field_generator = fields.plane_wave
        wavelengths = np.linspace(400, 1000, 5)
        kwargs = dict(theta=0, inc_angle=180)

        efield = fields.efield(field_generator,
                       wavelengths=wavelengths, kwargs=kwargs)
        n1 = n2 = 1.0
        dyads = propagators.DyadsQuasistatic123(n1=n1, n2=n2)

        sim = core.simulation(self.struct, efield, dyads)
        sim.scatter(verbose=False)
        field_kwargs = tools.get_possible_field_params_spectra(sim)

        config_idx = 0
        wl, spectrum = tools.calculate_spectrum(sim,
                            field_kwargs[config_idx], linear.extinct)
        
        abs_ = spectrum.T[2]/np.max(spectrum.T[2])
        
        return wl, abs_
        
    def _get_Iq(self, q, radius):
        """Obtain a SAS profile using Debye model(?)
        
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
        
    def get_saxs(self):
        """Obtain a SAS profile with polydispersity
        
        Implemented following the description provided in:
        https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html
        """
        q = np.logspace(np.log10(1e-3), np.log10(1), 200)
        radii = self.step*self.radii
        Iqs = [self._get_Iq(q, ri) for ri in radii]
        pdf = [self.radii_dist.pdf(ri) for ri in self.radii]
        integrand = np.asarray([Iqs[i]*pdf[i] for i in range(len(self.radii))])
        indx = np.argsort(radii)
        pq = np.asarray([np.trapz(integrand[indx,i], x = radii[indx]) for i in range(len(q))])
        
        # obtain the scale parameter
        V_mean = (4/3)*np.pi*((self.step*self.r_mu)**3)
        V_occupied = np.sum([(4/3)*np.pi*(ri**3) for ri in radii])
        lengths = self.geometry.max(axis=0) - self.geometry.min(axis=0)
        V_total = np.prod(lengths) 
        
        scale = V_occupied/(V_total*V_mean)
        
        return q, scale*pq
        
        
        
        
        
        
        
        