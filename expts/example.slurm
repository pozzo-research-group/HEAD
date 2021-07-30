#!/bin/bash

#SBATCH --job-name=example
#SBATCH --account=hpc
#SBATCH --partition=hpc
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=28
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --chdir=/gscratch/scrubbed/kiranvad/Genetic_Algorithm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --export=all
source env/bin/activate
python expts/example.py