# -----------------------------------------
# -  Input deck for realistic beam size   -
# -----------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_beam_g100_l200_d200_dyn_sub_psi0.01_1e5steps_3000x50_10Mpart"


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
npt_alpha = 3001  # number of mesh points along alpha
npt_chi = 51   # number of mesh points along chi
if (mpi_rank==0) : print("npt_alpha={}, npt_chi={}".format(npt_alpha, npt_chi))

####################### Main setup ##########################

## wavelet emission
num_wavefronts = 3000              # number of wavefronts
num_dirs = 50                      # number of field lines
num_step = 100000                  # number of steps (currently always equal to num_wavefronts)
dt = psi_max/num_step              # time step in electron rest frame
emission_interval = 30 #num_step-1     # only emit wavefronts at simulation end (test purpose)
emission_start = num_step-1 - (num_wavefronts-1)*emission_interval 

## remap
remap_interval = num_step-1        # interval of doing remapping (in time steps)
remap_scatter = False              # use scatter weights form for remap
remap_adaptive = False             # use adaptive smoothing length for remap
remap_scaling[0] = 1.0             # support/smoothing length scaling factor
remap_scaling[1] = 1.0             # support/smoothing length scaling factor
remap_verbose = False              # print remap statistics

# electron beam
beam_charge = 0.01                 # nC
num_particles = 160                # number of particles
trajectory_type = 2                # 1: straight line, 2: circular, 3: sinusoidal
parameters[0] = gamma              # central energy for all types
parameters[1] = 100.0              # propagation angle for type 1, radius (cm) for type 2, frequency for type 3

beam = init_beam(num_particles, gamma, lbeam, dbeam, mpi_rank)
beam[:,0] = beam[:,0] - 0.6*lbeam*1e-6
#beam = generate_microbunches(overall_beam_env='gaussian', _npart=num_particles, 
#                _nbunches=1, _sgmx_sub_div=6.0, _lbeam=lbeam, _dbeam=dbeam, _mpi_rank=mpi_rank)
# del beam     # init a single particle instead

## comoving mesh
num_gridpt_hor = npt_alpha                # number of points in x-axis
num_gridpt_ver = npt_chi                  # number of points in y-axis
mesh_span_angle = scaled_alpha/gamma**3   # in radians
mesh_width = scaled_chi/gamma**2          # in unit of radius

# load wavelets
load_subcycle_wavelets = True 
cosyr_root = '..'
path2subcycling = cosyr_root + "/input/wavelets/g100-200x200um-all_3000x50"
if (load_subcycle_wavelets) :
    wavelet_x, wavelet_y, wavelet_field = load_wavelets(path2subcycling, X_coord_file="scaled_xprime_all.npy",
        Y_coord_file="scaled_yprime_all.npy", fld_file="EBRad_all.npy", psi_file="Psi_all.npy", unscale_coord=True, 
        _gamma = gamma, broadcast=True, filter=True, filter_exp="np.where((psi<0.01) & (wx>-200.0/1e6))")
    if (mpi_rank==0) :
        print("wavelet shape =", wavelet_x.shape, wavelet_y.shape, wavelet_field.shape)
        print("wavelet field 0 min/max =", wavelet_field[0,:].min(), wavelet_field[0,:].max())
        print("wavelet field 1 min/max =", wavelet_field[1,:].min(), wavelet_field[1,:].max())
        print("wavelet field 2 min/max =", wavelet_field[2,:].min(), wavelet_field[2,:].max())

min_emit_angle = 0.0 

# 0: use global (x,y) coordinate; 
# 1: use local (x',y') coordinate; 
# 2: use local (alpha, chi) coordinate;
wavelet_type = 1   
if (wavelet_type == 0):
   rotation_angle = 0.1
   wavelet_x, wavelet_y = convert2global(wavelet_x, wavelet_y, rotation_angle) 

# True: loaded wavelets will be repeatedly emitted at each step and copied into internal wavelets array,
# otherwise only used when interpolation is done and not copied into internal wavelets array
use_wavelet_for_subcycle = True                               


####################### Diagnostics ##########################

print_interval = 100      # interval for printing simulation steps
beam_output_interval = num_step - 1
mesh_output_interval = num_step - 1
wavelet_output_interval = num_step - 1
beam_output = True

if (mpi_rank==0):
  make_output_dirs(run_name+"/beam", num_step, beam_output_start, beam_output_interval)
  mesh_output = True
  make_output_dirs(run_name+"/mesh", num_step, mesh_output_start, mesh_output_interval)
  wavelet_output = True 
  make_output_dirs(run_name+"/wavelet", num_step, wavelet_output_start, wavelet_output_interval)
  make_output_dirs(run_name+"/traj", num_step, wavelet_output_start, wavelet_output_interval)
