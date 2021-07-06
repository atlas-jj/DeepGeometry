#!/bin/bash
#SBATCH --job-name=tb
#SBATCH --account=def-jag
#SBATCH --time=6-23:07:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=output/%j.txt
#SBATCH --exclude=cdr2550
#SBATCH --mail-user=<jjin5@ualberta.ca>
#SBATCH --mail-type=ALL,TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"echo "Starting run at: `date`"
# ---------------------------------------------------------------------
source /home/jjin5/tensorflow/bin/activate
tensorboard --logdir=exp --host 0.0.0.0
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
