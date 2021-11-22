# High-throughput Experimentation using Active Design (HEAD)
This repository is a suite of tools to assist active design when using high-throughput expeimentation.
The goal is to develop off-the-shelf analysis tools for different material characterization combined with active learning, active search, Bayesian optimization, Genetic Algorithm and Reinforcement Learning.

We recommend installing the package in a editable format using the `install.sh` file in a Linux command line or using the instructions with-in the `install.sh` in a python command line.

## Examples and other details to be update soon


## Known issues
* sometimes, you may get the dreaded 'NotPSDError' from botorch/GPyTorch. We typically fix this by adjusting the noise prior on the likelihood in `gp_regression.py` file in the borth under MIN_INFERRED_NOISE_LEVEL = 5e-3.
see this [link](https://github.com/pytorch/botorch/issues/179#issuecomment-756462566) for more details.
