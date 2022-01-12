# Bayesian Optimization branch for HEAD
This branch contains code for the Bayesian Optimization part.

We recommend installing the package in a editable format using the `install.sh` file in a Linux command line or using the instructions with-in the `install.sh` in a python command line.

In a Linux command shell
```bash
chmod u+x install.sh 
./install.sh
```

## Example using the Gaussian function simulator can be performed as follows:

Import the required function using the below code snippet:
```python
from head import opentrons
import pandas as pd
import numpy as np
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")
```

We need a simulator to mimic a robotic experiment. We achieve this using the following:
```python
class Simulator:
    def __init__(self):
        self.domain = np.linspace(-5,5,num=100)
        
    def generate(self, mu, sig):
        scale = 1/(np.sqrt(2*np.pi)*sig)
        return scale*np.exp(-np.power(self.domain - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def process_batch(self, Cb, fname):
        out = []
        for c in Cb:
            out.append(self.generate(*c))
        out = np.asarray(out)
        df = pd.DataFrame(out.T, index=self.domain)
        df.to_excel(fname, engine='openpyxl')
        
        return 
    
    def make_target(self, ct):
        return self.domain, self.generate(*ct)

```
The simulator simply returns a Gaussian distribution function given a `mu` and `sigma` values. We use this to specify a target distribution:
```python
sim = Simulator()
target = np.array([-2,0.5])
xt, yt = sim.make_target(target)
```

Set up your design space using the lower and upper limits
```python
Cmu = [-5,5]
Csig = [0.1,3.5]
bounds = [Cmu, Csig]
```

Define a distance metric function
```python
import fdasrsf as fs

def APdist(point_a, point_b, time):
    curves = np.zeros((len(point_a), 2))
    curves[...,0] = point_a
    curves[...,1] = point_b
    obj = fs.fdawarp(curves, time)
    obj.srsf_align(parallel=True, MaxItr=50)
    dp = fs.efda_distance_phase(obj.qn[...,0], obj.qn[...,1])
    da = fs.efda_distance(obj.qn[...,0], obj.qn[...,1])

    return -(da+dp)

metric_function = lambda f1,f2 : APdist(f1,f2,xt)
```


Initiate the optimizer using the following:
```python
optim = opentrons.Optimizer(xt, yt, 
                            bounds, 
                            savedir = '../data',
                            batch_size=4,
                            metric = metric_function
                           )

```

Perform a random iteration
```python
# random iteration
optim.save()
C0 = np.load('../data/0/new_x.npy')
sim.process_batch(C0, '../data/opentrons/0.xlsx')
optim.update('../data/0.xlsx')
optim.save()
optim.get_current_best()
```

Perform the BO iterations with a specified budget
```python
for i in range(1,21):
    # iteration i selection
    optim.suggest_next()
    optim.save()
    # simulate iteration i new_x 
    Ci = np.load('../data/%d/new_x.npy'%i)
    sim.process_batch(Ci, '../data/%d.xlsx'%i)
    optim.update('../data/%d.xlsx'%i)
    optim.save()
    optim.get_current_best()

```

Note that when a robotic experiment is involved, each iteration has to  be performed with the robot in the loop thus we would perform the for loop one at a time. 
In a Jupyter notebook format, we would do this one iteration at a time, keeping the Kernel active and adding one new cell for each iteration below the previous iteration but performing same set of operations. A more neater approach for this is under the works.
At any given iteration, the function `get_current_best` reports what the algorithm thinks is a the best match so far.


