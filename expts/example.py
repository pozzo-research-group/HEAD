import numpy as np
import matplotlib.pyplot as plt
from head import Emulator, Grid
import os
import time

start = time.time()

savedir = '../figures/example/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

sim = Emulator()
X = np.linspace(3,5, num=2) 
Y = np.linspace(1,2, num=2)
grid = Grid(X,Y)

def run(r_mu, r_sigma, ind):
	fig = plt.figure(figsize=(12,3))
	# step 1: create the structure
	sim.make_structure(r_mu,r_sigma)
	ax = fig.add_subplot(1,3,1)
	sim.plot_structure2d(ax=ax)
	
	# step 2: simulate a absorption spectra
	wl, I = sim.get_spectrum()
	ax = fig.add_subplot(1,3,2)
	ax.plot(wl,I)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel('Intensity')

	# step 3: simulate SAS profile
	q, pq = sim.get_saxs()
	ax = fig.add_subplot(1,3,3)
	ax.loglog(q, pq)
	
	plt.savefig(savedir + '/%d.png'%ind)
	plt.close()

for i, point in enumerate(grid):
	run(point[0],point[1],i)

end = time.time()
print('Time elapsed : %.2f sec'%(end - start))
