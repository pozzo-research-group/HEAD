import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 15,
                     "savefig.dpi": 400,
                     "savefig.bbox": 'tight',
                     'text.latex.preamble': r'\usepackage{amsfonts}'
                    }
                   )

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.functions import SRVF

N_SAMPLES = 100
TARGET = [-2,0.5]

lambda_ = np.linspace(-5,5,num=N_SAMPLES)
Rn = Euclidean(N_SAMPLES)
srvf = SRVF(lambda_)
def gaussian(mu,sig):
    scale = 1/(np.sqrt(2*np.pi)*sig)
    return scale*np.exp(-np.power(lambda_ - mu, 2.) / (2 * np.power(sig, 2.)))

yt = gaussian(*TARGET)

dRn = lambda xi,yi : -float(Rn.metric.dist(yi, yt))

dSRSF = lambda xi,yi : -srvf.metric.dist(yi, yt)   

test_x = np.linspace(-5, 5, 51)
ground_truth_Rn = [dRn(i, gaussian(i,TARGET[1])) for i in test_x]
fig, ax = plt.subplots()
ax.plot(test_x, ground_truth_Rn , label=r'$\mathbb{R}^n$')
ground_truth_srvf = [dSRSF(i, gaussian(i,TARGET[1])) for i in test_x]

ax.plot(test_x, ground_truth_srvf, label='SRSF')

ax.set_xlabel(r'$\mathcal{X}$')
ax.set_ylabel(r'$d_{\mathcal{M}}(x, x_{t})$')
ax.legend()
plt.savefig('mse_vs_srsf.png')
plt.close()