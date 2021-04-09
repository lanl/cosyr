#!/bin/bash

#SBATCH -N 360
#SBATCH -A m3757
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J csr
#SBATCH -t 00:30:00
#SBATCH --tasks-per-node=64

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

build="$(pwd)/../build"
test="test_beam_g100_l200_sub_10000x100"

cd ${build} && srun --cpu_bind=cores ./cosyr ../input/${test}.py
