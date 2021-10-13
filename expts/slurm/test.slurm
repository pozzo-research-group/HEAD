#!/bin/bash

#SBATCH --job-name=test
#SBATCH --account=genpool-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --output=./joblogs/pred_test.out
#SBATCH --error=./joblogs/err_test.out
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
python -u ./scripts/tests.py > ./joblogs/tests.out
exit