#!/bin/bash

#SBATCH --job-name=example
#SBATCH --account=genpool-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --output=pred.out
#SBATCH --error=err.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

cd /usr/lusers/kiranvad/scratch/Genetic_Algorithm/
source env/bin/activate
which python
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
pwd
python -u scripts/data_genetator_uvvis.py > output.out
exit