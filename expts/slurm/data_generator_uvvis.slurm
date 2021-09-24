#!/bin/bash

#SBATCH --job-name=datagen
#SBATCH --account=genpool
#SBATCH --partition=hpc
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=12
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --output=./joblogs/pred_datagen.out
#SBATCH --error=./joblogs/err_datagen.out
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all
#SBATCH --exclusive

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
python -u ./scripts/data_genetator_uvvis.py > ./joblogs/output_datagen.out
exit



