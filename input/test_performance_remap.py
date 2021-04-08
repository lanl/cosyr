# ----------------------------------------------
# -  Input deck for interpolation performance  -
# ----------------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_performance_remap"

## electron and trajectory
gamma=1000
# Ct=2  
delta_alpha_B = 0.2  # assume the bending section spreads over \Delta\alpha=0.2
lbeam = 3000        # um
dbeam = 50          # um
delta_alpha_tilde = 10000 #gamma * delta_alpha_B / 2
delta_chi_tilde = dbeam * 1e-6 * gamma**2.0

if (mpi_rank==0) : print("delta_alpha_tilde={}, delta_chi_tilde={}".format(delta_alpha_tilde, delta_chi_tilde))

mesh_alpha_npt = 4000 #int(delta_alpha_tilde/1.0+1.0)
mesh_chi_npt = 4000 #int(delta_chi_tilde/1.0+1.0)

if not mesh_alpha_npt%2:
    mesh_alpha_npt += 1
if not mesh_chi_npt%2:
    mesh_chi_npt += 1

if (mpi_rank==0) : print("mesh_alpha_npt={}, mesh_chi_npt={}".format(mesh_alpha_npt, mesh_chi_npt))

Nw = 2000

####################### Main setup ##########################

# wavelet emission 
num_wavefronts = 200               # number of wavefronts
num_dirs = 0                       # number of field lines
num_step = num_wavefronts          # number of steps
dt = delta_alpha_B/num_wavefronts  # time step in electron rest frame
emission_interval = num_step - 1   # only emit wavefronts at simulation end (test purpose)
# dt=Ct/(gamma*Nw)

## remap
remap_interval = num_step - 1      # interval of doing remapping (in time steps)
remap_scatter = False              # use scatter weights form for remap
remap_adaptive = False             # use adaptive smoothing length for remap
remap_scaling[0] = 0.2             # support/smoothing length scaling factor
remap_scaling[1] = 0.2             # support/smoothing length scaling factor
remap_verbose = False              # print remap statistics

# --- eletron beam info ---
beam_charge = 0.01                 # nC
num_particles = 10                 # number of particles
trajectory_type = 2                # 1: straight line, 2: circular, 3: sinusoidal
parameters[0] = gamma              # parameters for describing the motion of electron
parameters[1] = 100.0

beam = init_beam(num_particles, gamma, lbeam, dbeam, mpi_rank)
del beam                           # init a single particle instead

# --- comoving mesh info ---
num_gridpt_hor = mesh_alpha_npt                # number of points in x-axis
num_gridpt_ver = mesh_chi_npt                  # number of points in y-axis
mesh_span_angle = delta_alpha_tilde/gamma**3   # in radians
mesh_width = delta_chi_tilde/gamma**2          # in unit of radius

# --- wavelets ---
#------- generate regular mesh points and used as wavelets
x = np.linspace(-delta_alpha_tilde/2.0, delta_alpha_tilde/2.0, np.int(mesh_alpha_npt*1))
y = np.linspace(-delta_chi_tilde/2.0, delta_chi_tilde/2.0, np.int(mesh_chi_npt*1))
ymax = y.max()
ymin = y.min()
wavelet_x, wavelet_y = np.meshgrid(x,y) 
wavelet_x = wavelet_x.flatten()
wavelet_y = wavelet_y.flatten()
wavelet_field = np.cos(5.0*wavelet_y*2*np.pi/(ymax-ymin))
num_wavelet_fields = 1

# unscale if necessary
wavelet_x /= gamma**3.0
wavelet_y /= gamma**2.0
wavelet_field *= gamma**4.0

min_emit_angle = 10.0

wavelet_type = 1   # 0: use global (x,y) coordinate; 1: use local (x',y') coordinate; 2: (TODO) use local cylindrical coordinate
if (wavelet_type == 0) :
   rotation_angle = 0.1
   wavelet_x, wavelet_y = convert2global(wavelet_x, wavelet_y, rotation_angle) 

use_wavelet_for_subcycle = False # True: loaded wavelets will be repeatedly emitted at each step and copied into internal wavelets array, otherwise only used when interpolation is done and not copied into internal wavelets array
num_wavelet_fields = 1

####################### Diagnostics ##########################

print_interval = 100        # interval for printing simulation steps
beam_output_interval = num_step - 1
mesh_output_interval = num_step - 1
wavelet_output_interval = num_step - 1
beam_output = False

if (mpi_rank==0):
  make_output_dirs(run_name+"/beam", num_step, beam_output_start, beam_output_interval)
  mesh_output = True
  make_output_dirs(run_name+"/mesh", num_step, mesh_output_start, mesh_output_interval)
  wavelet_output = False
  make_output_dirs(run_name+"/wavelet", num_step, wavelet_output_start, wavelet_output_interval)
  make_output_dirs(run_name+"/traj", num_step, wavelet_output_start, wavelet_output_interval)

