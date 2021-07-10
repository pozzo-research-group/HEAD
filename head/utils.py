from pyGDM2 import (structures, materials, core, 
                    linear, fields, propagators, 
                    tools
                   )
import head
import numpy as np
from modAL.models import BayesianOptimizer
import matplotlib.pyplot as plt


def get_spectrum(l, r):
    """ Obtain a simulated absorption spectra for a hexagonal nanorod mesh
    L -- length of the cylinder
    R -- radius of the cylinder

    returns the spectrum as a head.UVVis object
    """
    step = 20
    geometry = structures.nanorod(step, L=l, R=r, mesh='hex')
    material = materials.gold()
    struct = structures.struct(step, geometry, material, verbose=False)
    field_generator = fields.plane_wave
    wavelengths = np.linspace(400, 1000, 50)
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
    
    return head.UVVis(wl, abs_)


class ExampleRunnerSimulation:
    def __init__(self, dspace):
        self.dspace = dspace
        self.Lt, self.Rt = 2, 1
        self.st = get_spectrum(self.Lt,self.Rt)
    
    def _get_length_radius(self,p):
        L = p[0]**2
        R = 0.5*(L/p[1])
        
        return [L,R]

    def _oracle(self,query, measure):
        pi = self._get_length_radius(query)
        si = get_spectrum(pi[0],pi[1])
        yi = measure(si, self.st)

        return yi

    def iterate(self, model,query_strategy, measure,
                n_iter=5, n_init=5, b=5):
        self.oracle = lambda x: self._oracle(x, measure)
        X0 = self.request_batch(n_init, is_init=True)
        Y0,_ = self.evaluate_batch(X0)
        
        self.optimizer = BayesianOptimizer(
            estimator=model,
            query_strategy=query_strategy,
            X_training = X0,
            y_training = Y0
        )
        
        self.loss_evol = []
        for i in range(n_iter):
            Xbi = self.request_batch(b)
            Ybi,losses = self.evaluate_batch(Xbi)
            [self.loss_evol.append(l) for l in losses]
            self.optimizer.teach(Xbi, Ybi)

    def request_batch(self,b, is_init=False):
        if is_init:
            Xb = self.dspace.sample(n_samples=b)
        else:
            query_idx, _ = self.optimizer.query(self.dspace.space, b=b)
            Xb = self.dspace[query_idx,:]
            
        return np.asarray(Xb)
    
    def evaluate_batch(self, Xb):
        losses, Yb = [], []
        for xi in Xb:
            yi = self.oracle(xi)
            Yb.append(yi)
            losses.append(yi)

        return np.asarray(Yb), losses
    
    def plot_bestmatch(self):
        X_max, y_max = self.optimizer.get_max()
        Lbest, Rbest = self._get_length_radius(X_max)
        sbest = get_spectrum(Lbest,Rbest)
        fig, ax = plt.subplots()
        sbest.plot(ax)
        self.st.plot(ax)
        ax.set_title('Oracle score {:.2f}'.format(float(y_max)))
        ax.legend(['Best [%.2f, %.2f]'%(Lbest,Rbest), 'Target [%.2f, %.2f]'%(self.Lt,self.Rt)])
        plt.show()
        
    def plot_learning(self):
        y_pred, y_std = self.optimizer.predict(self.dspace.space, return_std=True)
        X_max, y_max = self.optimizer.get_max()
        fig, axs = plt.subplots(1,2,figsize=(2*4*1.6,4))
        im = axs[0].contourf(self.dspace.mesh[0], self.dspace.mesh[1], 
                         y_pred.reshape(10,10), cmap='RdYlGn')
        fig.colorbar(im, ax=axs[0])

        axs[0].set_xlabel(r'$\sqrt{L}$')
        axs[0].set_ylabel(r'$L/D$')
        axs[0].scatter(self.optimizer.X_training[:,0], self.optimizer.X_training[:,1], 
                   s=25, color='k', label='Queried')
        axs[0].scatter(X_max[0],X_max[1],s=100,marker='*',label='best',color='tab:blue',)
        axs[0].scatter(np.sqrt(self.Lt),self.Lt/(2*self.Rt),s=100,color='tab:red',marker='*',label='Target')
        axs[0].legend()
        
        axs[1].scatter(np.arange(len(self.loss_evol)), self.loss_evol)
        axs[1].set_xlabel('Query Index')
        axs[1].set_ylabel('Oracle value')
                       
        for ax in axs:
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
        plt.show()