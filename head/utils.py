from pyGDM2 import (structures, materials, core, 
                    linear, fields, propagators, 
                    tools
                   )
import head
import numpy as np
from modAL.models import BayesianOptimizer
import matplotlib.pyplot as plt
import pdb

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
    wavelengths = np.linspace(400, 1000, 100)
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
    
    try:
        return head.UVVis(wl, abs_)
    except:
        return wl,abs_


class ExampleRunner:
    def __init__(self,dspace):
        """Example Bayesian Optimizer for UVVis experiments

        Inputs:
            dspace : one of the implemented `head.datareps` objects

        """
        self.dspace = dspace
        self.S_training = []
        self.optimizer = {}

    def add_target(self,st, **kwargs):
        """Add a target spectrum
        """
        self.st = st
        for key,value in kwargs.items():
            setattr(self,'target_'+key,value)

    def oracle(self,s):
        """Score function for spectrum

        A function that takes a spectrum in s and returns its score relative 
        to a user-defined measure with self.st the target (spectrum)
        """
        raise NotImplementedError

    def get_best_spectrum(self,x):
        """Obtain the best spectrum from the trained data
        inputs:
            x : concentration array (a point in self.dspace)
        """

        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(self.optimizer.X_training)
        x = np.asarray(x).reshape(1,len(x))
        _, ind = neigh.kneighbors(x, n_neighbors=1)
        S_training = [item for sublist in self.S_training for item in sublist]

        return S_training[int(ind)]

    def request_batch(self,b, is_init=False):
        """
        Request a batch of samples to be evaluated
        inputs:
            b : batch size (int)
            is_init : whether the request batch is an initial set of samples from self.dspace (boolean, False)
        """
        if is_init:
            Xb = self.dspace.sample(n_samples=b)
        else:
            query_idx, _ = self.optimizer.query(self.dspace.points, b=b)
            Xb = self.dspace[query_idx,:]
            
        return np.asarray(Xb)
    
    def evaluate_batch(self, Sb):
        """
        Given a set of spectra as a list indexed by the request batch, return their loss/scores

        inputs:
            Sb : list of head.UVVis objects indexed by the corresponding batch concentrations
        """
        self.S_training.append(Sb)
        print(len(self.S_training))
        Yb = []
        for si in Sb:
            yi = self.oracle(si, self.st)
            Yb.append(yi)

        return np.asarray(Yb)

    def set_params(self,**params):
        """ Utility method to add functions to the class
        """
        for key,value in params.items():
            setattr(self,key,value)

        return self

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
            query_idx, _ = self.optimizer.query(self.dspace.points, b=b)
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
        y_pred, y_std = self.optimizer.predict(self.dspace.points, return_std=True)
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