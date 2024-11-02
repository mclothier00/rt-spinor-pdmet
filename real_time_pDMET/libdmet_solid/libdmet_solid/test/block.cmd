#!/bin/bash

#SBATCH --partition=debug,smallmem,parallel,serial
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH -c 28
#SBATCH -t 1:00:00
#SBATCH --mem=120000

export TMPDIR="/scratch/global/zhcui/"
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

module load gcc-5.4.0/boost-1.55.0-openmpi-1.10.3 

source /home/zhcui/.bashrc_libdmet_model

srun hostname
ulimit -l unlimited
python ./test_hub2dbcs_nib.py

