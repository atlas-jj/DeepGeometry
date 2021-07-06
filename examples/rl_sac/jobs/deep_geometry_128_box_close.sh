#!/bin/bash
#SBATCH --job-name=128_m1_deep_geometry
#SBATCH --account=def-jag
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=6-23:07:00
#SBATCH --mem-per-cpu=80000M
#SBATCH --output=output/%j.txt
#SBATCH --exclude=cdr2550
#SBATCH --mail-user=<jjin5@ualberta.ca>
#SBATCH --mail-type=ALL,TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"echo "Starting run at: `date`"
# ---------------------------------------------------------------------
source /home/jjin5/env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jjin5/.mujoco/mujoco200/bin
python train_meta_world.py taskid=4 agent=m1_deep_geometry128 headless=True experiment=meta_box_close128
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
