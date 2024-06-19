#!/bin/bash

## This is a Slurm job script for A1

#SBATCH --job-name=iom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=xs-4114
#SBATCH --mem=1gb
#SBATCH --time=00:10:00
#SBATCH --output=iom_%j.slurmlog
#SBATCH --error=iom_%j.slurmlog
#SBATCH --mail-type=NONE

echo "Running iom job!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
echo "Arguments to your executable: $@"

# Runs your script with the arguments you passed in
echo "Running iom now..."
srun perf stat -r 3 -- ./iom_cpp.out $@ 


echo "Job ended at $(date)"
