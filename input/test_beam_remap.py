# -------------------------------------------------------------
# -  Input deck for interpolation test with far field approx. -
# -------------------------------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_beam_remap_g1000"

## electron and trajectory
gamma=1000
l_beam = 200 #3000   # beam length, in um
d_beam = 200 #50     # beam radius, in um
psi_max = 1e-4 #0.42   # max retarded 

## common mesh 
scaled_alpha = l_beam*1e-6*gamma**3.0 * 3.0 # scaled alpha range of mesh 
scaled_chi = d_beam*1e-6*gamma**2.0 * 2.0 # scaled chi range of mesh
if (mpi_rank==0) : print("scaled_alpha={}, scaled_chi={}".format(scaled_alpha, scaled_chi))
npt_alpha = 1001 #1001  # number of mesh points along alpha
npt_chi = 3  #101   # number of mesh points along chi
wpc = [1,1]         # wavlets per cell
if (mpi_rank==0) : print("npt_alpha={}, npt_chi={}".format(npt_alpha, npt_chi))

####################### Main setup ##########################

## wavelet emission
num_wavefronts = 2                 # number of wavefronts
num_dirs = 0                       # number of field lines
num_step = num_wavefronts          # number of steps (currently always equal to num_wavefronts)
dt = psi_max/num_wavefronts        # time step in electron rest frame
emission_interval = num_step - 1   # only emit wavefronts at simulation end (test purpose)

## remap
remap_interval = num_step - 1       # interval of doing remapping (in time steps)
remap_scatter = False               # use scatter weights form for remap
remap_adaptive = False              # use adaptive smoothing length for remap
remap_scaling[0] = 1.0              # base support/smoothing length scaling factor
remap_scaling[1] = 1.0              # base support/smoothing length scaling factor
remap_verbose = False               # print remap statistics

# electron beam
beam_charge = 0.01                 # nC
num_particles = 3000             # number of particles
trajectory_type = 2                # 1: straight line, 2: circular, 3: sinusoidal
parameters[0] = gamma              # central energy for all types
parameters[1] = 100.0              # propagation angle for type 1, radius (cm) for type 2, frequency for type 3

beam = init_1D_beam(num_particles, gamma, l_beam, d_beam, mpi_rank)
# del beam                           # init a single particle instead

## comoving mesh
num_gridpt_hor = npt_alpha                # number of points in x-axis
num_gridpt_ver = npt_chi                  # number of points in y-axis
mesh_span_angle = scaled_alpha/gamma**3   # in radians
mesh_width = scaled_chi/gamma**2          # in unit of radius

## load wavelets
wavelet_x, wavelet_y, wavelet_field = gen_test_wavelets(scaled_alpha*2, scaled_chi*2, np.int((npt_alpha-1)*wpc[0]*2+2), np.int((npt_chi-1)*wpc[1]*2+1), gamma, unscale_coord=True, distribution="adaptive_1d", method="potential", filter=True, filter_exp="np.abs(wx) > 300.0")
if (mpi_rank==0) :
    print("wavelet shape =", wavelet_x.shape, wavelet_y.shape, wavelet_field.shape)
    if wavelet_field.ndim == 1:
      print("wavelet field min/max =", wavelet_field.min(), wavelet_field.max())
    else :   
      for i in range(wavelet_field.shape[0]) :
        print("wavelet field {1} min/max =".format(i), wavelet_field[i,:].min(), wavelet_field[i,:].max())

min_emit_angle = 0.0 

# 0: use global (x,y) coordinate; 
# 1: use local (x',y') coordinate; 
# 2: (TODO) use local cylindrical coordinate
wavelet_type = 1   
if (wavelet_type == 0):
   rotation_angle = 0.1
   wavelet_x, wavelet_y = convert2global(wavelet_x, wavelet_y, rotation_angle) 

# True: loaded wavelets will be repeatedly emitted at each step and copied into internal wavelets array,
# otherwise only used when interpolation is done and not copied into internal wavelets array
use_wavelet_for_subcycle = True
num_wavelet_fields = 1

####################### Diagnostics ##########################

print_interval = 1        # interval for printing simulation steps
beam_output_interval = 1 
mesh_output_interval = 1
wavelet_output_interval = 1
beam_output = True

if (mpi_rank==0):
  make_output_dirs(run_name+"/beam", num_step, beam_output_start, beam_output_interval)
  mesh_output = True
  make_output_dirs(run_name+"/mesh", num_step, mesh_output_start, mesh_output_interval)
  wavelet_output = True
  make_output_dirs(run_name+"/wavelet", num_step, wavelet_output_start, wavelet_output_interval)
  make_output_dirs(run_name+"/traj", num_step, wavelet_output_start, wavelet_output_interval)
