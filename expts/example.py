import numpy as np
import matplotlib.pyplot as plt
from head import Emulator

import time

start = time.time()


sim = Emulator(r_mu=5)

fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(1,3,1)
sim.plot_structure2d(ax=ax)

wl, I = sim.get_spectrum()
ax = fig.add_subplot(1,3,2)
ax.plot(wl,I)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity')

q, pq = sim.get_saxs()
ax = fig.add_subplot(1,3,3)
ax.loglog(q, pq)
plt.savefig('../figures/1.png')

end = time.time()
print('Time elapsed : %.2f'%(end - start))
