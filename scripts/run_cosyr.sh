
export OMP_NUM_THREADS=1
#export OMP_PLACES=cores
#export OMP_PLACES=threads
#export OMP_PROC_BIND=close
#export OMP_PROC_BIND=spread
#export OMP_PROC_BIND=true
#export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
#export KOKKOS_NUM_DEVICES=2
# export PYTHONPATH=/Users/huangck/tmp/h5py-mpi/venv37/lib/python3.7/site-packages

if [ ! -d "./data" ]
then 
    echo "Creating data folder"
    mkdir ./data
else
    rm -rf ./data/*.dat
fi

#mpirun -np 2 --bind-to socket --report-bindings ./build/cosyr input/test_remap_eval.py 
#mpirun -np 2 --map-by ppr:2:socket --display-map ./build/cosyr input/test_remap_eval.py 
#mpirun -np 2 --npersocket 1 --bind-to socket --display-map ./build/cosyr input/test_kernel_performance.py 
#mpirun -np 1 --display-map ./build/cosyr input/test_kernel_performance.py 
#--kokkos-threads=8 
#mpirun -np 1 ./build/cosyr input/test_interpolation_performance.py --kokkos-threads=10 --num-devices=1 
mpirun -np 1 --npersocket 1 --bind-to socket --display-map ./build/cosyr input/test_interpolation_performance.py 
