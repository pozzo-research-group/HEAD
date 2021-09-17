#!/bin/bash

#SBATCH --job-name=MOO_SUCB
#SBATCH --account=genpool-ckpt
#SBATCH --partition=ckpt-gpu
#SBATCH --gres=gpu:P100:1
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/gscratch/scrubbed/kiranvad/Genetic_Algorithm
#SBATCH --output=expts/joblogs/pred_MOO_SUCB.out
#SBATCH --error=expts/joblogs/err_MOO_SUCB.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "Environment from : "$SLURM_CHDIR
source env/bin/activate
which python
module load cuda/11.1.1-1
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
python MOO_SUCB.py
exit