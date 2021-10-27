# Bayesian Optimization branch for HEAD
This branch contains code for the Bayesian Optimization part.

We recommend installing the package in a editable format using the `install.sh` file in a Linux command line or using the instructions with-in the `install.sh` in a python command line.

## Example using the Gaussian function simulator can be performed as follows:
1. Import the required function using the below code snippet:
```python
from head import opentrons
import pandas as pd
import numpy as np
from scipy.spatial import distance
```

2. We need a simulator to mimic a robotic experiment. We achieve this using the following:
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
3. The simulator simply returns a Gaussian distribution function given a `mu` and `sigma` values. We use this to specify a target distribution:
```python
sim = Simulator()
target = np.array([-2,0.5])
xt, yt = sim.make_target(target)
```

4. Set up your design space using the lower and upper limits
```python
Cmu = [-5,5]
Csig = [0.1,3.5]
bounds = [Cmu, Csig]
```

5. Initiate the optimizer using the following:
```python
optim = opentrons.Optimizer(xt, yt, bounds, 
                            savedir = '../data/', batch_size=4)

```

6. Perform a random iteration
```python
# random iteration
optim.save()
C0 = np.load('../data/0/new_x.npy')
sim.process_batch(C0, '../data/opentrons/0.xlsx')
optim.update('../data/0.xlsx')
optim.save()

```
7. Perform the BO iterations with a specified budget
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

```

Note that when a robotic experiment is involved, each iteration has to  be performed with the robot in the loop thus we would perform the for loop one at a time. 
In a Jupyter notebook format, we would do this one iteration at a time, keeping the Kernel active and adding one new cell for each iteration below the previous iteration but performing same set of operations. A more neater approach for this is under the works.

