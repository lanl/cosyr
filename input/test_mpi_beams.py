# ---------------------------------------------------------
# -  Input deck for beam hdf output (through h5py/mpi4py) -
# ---------------------------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_mpi_beams"

#the main code pass the MPI rank in variable mpi_rank
print("my rank from main = ", mpi_rank)

#we can also get MPI rank and communicator from mpi4py, the rank should be the same as in the main code
#from mpi4py import MPI
#print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank() + ", and my rank from main = ", mpi_rank)

## electron and trajectory
gamma=100
lbeam = 200   # beam length, in um
dbeam = 200     # beam diameter, in um
psi_max = 0.3   # max retarded 

## common mesh
box_beam_ratio = 2.0  # mesh size / beam size 
scaled_alpha = lbeam*1e-6*gamma**3.0 * box_beam_ratio # scaled alpha range of mesh 
scaled_chi   = dbeam*1e-6*gamma**2.0 * box_beam_ratio # scaled chi range of mesh
if (mpi_rank==0) : print("scaled_alpha={}, scaled_chi={}".format(scaled_alpha, scaled_chi))
npt_alpha = 31  # number of mesh points along alpha
npt_chi = 51   # number of mesh points along chi
if (mpi_rank==0) : print("npt_alpha={}, npt_chi={}".format(npt_alpha, npt_chi))

####################### Main setup ##########################

## wavelet emission
num_wavefronts = 1                 # number of wavefronts
num_dirs = 0                       # number of field lines
num_step = num_wavefronts          # number of steps (currently always equal to num_wavefronts)
dt = psi_max/num_wavefronts        # time step in electron rest frame
emission_interval = num_step - 1   # only emit wavefronts at simulation end (test purpose)

## remap
remap_interval = num_step - 1      # interval of doing remapping (in time steps)
remap_scatter = False              # use scatter weights form for remap
remap_adaptive = False             # use adaptive smoothing length for remap
remap_scaling[0] = 0.5             # support/smoothing length scaling factor
remap_scaling[1] = 0.5             # support/smoothing length scaling factor
remap_verbose = False              # print remap statistics

# electron beam
beam_charge = 0.01                 # nC
num_particles = 10                 # number of particles
trajectory_type = 2                # 1: straight line, 2: circular, 3: sinusoidal
parameters[0] = gamma              # central energy for all types
parameters[1] = 100.0              # propagation angle for type 1, radius (cm) for type 2, frequency for type 3

beam = init_beam(num_particles, gamma, lbeam, dbeam, mpi_rank)

#print(beam)

beam_output = True
if (mpi_rank==0):
  make_output_dirs(run_name+"/beam", num_step, beam_output_start, beam_output_interval)

write_beam(beam, run_name+"/beam/0", mpi_rank)
# parallel hdf5 output require h5py with parallel support and mpi4py
#parallel_write_beam(beam, num_particles, run_name+"/beam/0", mpi_rank)

# Delete beam if we want to use the beam initialization in the main code
#del beam

