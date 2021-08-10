#!/bin/bash

#SBATCH --job-name=example
#SBATCH --account=genpool
#SBATCH --partition=hpc
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=28
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --chdir=/gscratch/scrubbed/kiranvad/Genetic_Algorithm
#SBATCH --output=expts/pred.out
#SBATCH --error=expts/err.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

source env/bin/activate
which python
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
python example.py
exit