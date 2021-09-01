Perfoming optimization with a a robot requires users to periodically collect data and update models.
This directory shows an example of performing multi-objective optimization using Bayesian Opitmization techniques to achieve a target shape spectra (SAS and UV-Vis in this case) with in a grid design space comprising of mean and variance of log-normal distribution for the radii of spehres.
The geometry is simulated in pyGDM2 for uv-vis and an analytical formula from sasmodels package is used for SAXS profile. 
The modelling part is implemented in the `head/modelling.py`

Use the following protocol to perform a BO run:

All the parameters are set with in the `config.ini` file. The variables names there are self-explanatory.


1. Initiate the optimization by randomly sampling in a grid
```bash
source env/bin/activate
cd OT2
python3 initialize.py
python3 generate_random_batch.py
python3 ot2_platereader.py
python3 teach_bo.py
```
2. Run a single BO iteration using the following
```bash
python3 run_bo.py
python3 ot2_platereader.py 
python3 teach_bo.py

```

In practice, you would implemente a `ot2_platereader.py` file that simply converts the data you collected into .txt files of a specific format into folders named `spectra_i` where `i` is the iteration containing `b` batch of samples. 

3. There's a helper functions that can be used to create some plots but this mainly works with a simulator for now.
```bash
python3 make_plots.py

```

4. If you have exhausted the number of iterations in `n_iterations`, running `run_bo.py` would return the current best sample from the optimization