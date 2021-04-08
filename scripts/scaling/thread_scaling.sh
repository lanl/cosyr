#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -q standard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J csr

set -e

# build
cd ..
source ./build -c kodiak

# run
run() {
  echo " ----------------- "
  echo " - num_threads: $1 "
  echo " ----------------- "

  input_deck="../input/test_interpolation_performance.py"
  export OMP_NUM_THREADS=$1
  mpirun -np 1 --bind-to socket --report-bindings ./cosyr ${input_deck}
}

use_hyperthreading=false
num_cores=36
last_step=1

if ${use_hyperthreading}; then
  num_max_threads=$(2 * num_cores)
  export OMP_PROC_BIND=close OMP_PLACES=threads
  step=4
else
  num_max_threads=${num_cores}
  export OMP_PROC_BIND=close OMP_PLACES=cores
  step=2
fi

for ((i=1; i <= ${num_max_threads}; i*=${step})); do
  run $i
  last_step=$i
done

if [ $last_step -ne $num_max_threads ]; then
  run $num_max_threads
fi
