#!/bin/bash

#SBATCH --job-name=SMBO
#SBATCH --account=genpool
#SBATCH --partition=hpc
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./joblogs/pred_SMBO.out
#SBATCH --error=./joblogs/err_SMBO.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

source ../env/bin/activate
echo "python from the following virtualenv"
which python
echo "working directory = "
pwd
ulimit -s unlimited
python -u scripts/shape_matching.py
exit
