import numpy as np
import matplotlib.pyplot as plt
from head import Emulator, Grid
import os, shutil
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()

savedir = '../figures/example/'
if  os.path.exists(savedir):
	shutil.rmtree(savedir)
os.makedirs(savedir)

X = np.linspace(10,40, num=10) 
Y = np.linspace(0.01,1, num=10)
grid = Grid(X,Y)
fig, ax = plt.subplots()
ax.scatter(grid.points[:,0], grid.points[:,1])
ax.set_xlabel(r'$r_{\mu}$')
ax.set_ylabel(r'$r_{\sigma}$')
plt.show()


sim = Emulator(use_mean=False)

def run(mu, sigma, ind):
    fig, axs = plt.subplots(1,3,figsize=(4*3,4))
	fig.subplots_adjust(wspace=0.5)
    sim.make_structure(r_mu=mu,r_sigma=sigma)
    sim.plot_radii(axs[0])
    axs[0].set_xlabel('radius')
    axs[0].set_ylabel('PDF')

    q, pq = sim.get_saxs(n_samples=100)
    axs[1].plot(q, pq)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    plt.setp(axs[1], xlabel='q (1/A)', ylabel='I(q)')

    wl, abs_ = sim.get_spectrum(n_samples=100)
    axs[2].plot(wl, abs_)
    plt.setp(axs[2], xlabel=r'$\lambda$ (nm)', ylabel='abs (a.u.)')

	fig.suptitle('r = '+','.join('%.2f'%i for i in sim.step*sim.radii))
	plt.savefig(savedir + '/%d.png'%ind, bbox_inches='tight')
	plt.close()

for i, point in enumerate(grid):
	run(point[0],point[1],i)

end = time.time()
print('Time elapsed : %.2f sec'%(end - start))
