#!/bin/bash

#SBATCH --job-name=BO_SUCB
#SBATCH --account=genpool
#SBATCH --partition=hpc
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=./joblogs/pred_BO_SUCB.out
#SBATCH --error=./joblogs/err_BO_SUCB.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

module load cuda/11.1.1-1
source ../env/bin/activate
echo "python from the following virtualenv"
which python
echo "working directory = "
pwd
ulimit -s unlimited
python -u scripts/BO_SUCB.py > ./joblogs/output_BO_SUCB.out
exit
