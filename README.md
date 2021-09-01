# High-throughput Experimentation using Active Design (HEAD)
This repository is a suite of tools to assist active design when using high-throughput expeimentation.
The goal is to develop off-the-shelf analysis tools for different material characterization combined with active learning, active search, Bayesian optimization, Genetic Algorithm and Reinforcement Learning.

We recommend installing the package in a editable format using the `install.sh` file in a Linux command line or using the instructions with-in the `install.sh` in a python command line.

## 1. Bayesian Optimization using batched HTE
You can find an example case study in the `OT2/` directory detailing how to perform a Multi-Objective Bayesian optimization using `botorch`.

This example case study is heavily based on the botorch tutorials but tested for a simple uv-vis (based on `pyGDM2`) and SAXS simulators (based on `sasmodels`). 