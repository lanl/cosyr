#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -p knl-quad_cache
#SBATCH -N 6
#SBATCH -n 1
#SBATCH -J csr

# build
cd ..
source ./build -c darwin -l intel

# run
run() {
  echo " ----------------- "
  echo " - num_ranks: $1 "
  echo " ----------------- "

  input_deck="../input/test_kernel_performance.py"
  export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
  mpirun -np $1 --bind-to socket --report-bindings ./cosyr ${input_deck}
}

num_max_ranks=6

for ((i=1; i <= ${num_max_ranks}; i++)); do
  run $i
done
