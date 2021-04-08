# -------------------------------------------------
# -  Input deck for wavelets with analytic field  -
# -------------------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_subcycle_wavelets"

debug = True 
gamma=1000
delta_alpha_B = 0.2  # assume the bending section spreads over \Delta\alpha=0.2
lbeam = 3000        # um
dbeam = 100         # um
delta_alpha_tilde = 10000 #gamma * delta_alpha_B / 2
delta_chi_tilde = dbeam * 1e-6 * gamma**2.0

if (mpi_rank==0) : print("delta_alpha_tilde={}, delta_chi_tilde={}".format(delta_alpha_tilde, delta_chi_tilde))

mesh_alpha_npt = 100 #int(delta_alpha_tilde/1.0+1.0)
mesh_chi_npt = 100 #int(delta_chi_tilde/1.0+1.0)

if not mesh_alpha_npt%2:
    mesh_alpha_npt += 1
if not mesh_chi_npt%2:
    mesh_chi_npt += 1

if (mpi_rank==0) : print("mesh_alpha_npt={}, mesh_chi_npt={}".format(mesh_alpha_npt, mesh_chi_npt))

####################### Main setup ##########################

# --- kernel mesh info ---
num_wavefronts = 2                 # number of wavefronts
num_dirs = 0                       # number of field lines
num_step = num_wavefronts          # number of steps
dt = delta_alpha_B/num_wavefronts  # time step in electron rest frame
emission_interval = num_step - 1   # only emit wavefronts at simulation end (test purpose)

## remap
remap_interval = num_step - 1      # interval of doing remapping (in time steps)
remap_scatter = False              # use scatter weights form for remap
remap_scaling[0] = 1.0             # support/smoothing length scaling factor
remap_scaling[1] = 1.0             # support/smoothing length scaling factor

# --- eletron beam info ---
num_particles = 1                  # number of particles
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
#path = "./input/wavelets/analytic_test/waveletsAdapativeLeft.csv"
path = "../input/wavelets/analytic_test/wavelets_adaptative.csv"

def field_func(x, y, yscale) :
    import numpy as np
    ymax = y.max()
    ymin = y.min()
    print("ymin = {}, ymax = {}".format(ymin, ymax))
    return np.cos(yscale*y*2*np.pi/(ymax-ymin))   # test case, assign test field of cos(x)

wavelet_x, wavelet_y, wavelet_field = load_wavelets_with_analytic_field(path, True, True, gamma, field_func, 5.0)

wavelet_type = 1   # 0: use global (x,y) coordinate; 1: use local (x',y') coordinate; 2: (TODO) use local cylindrical coordinate
use_wavelet_for_subcycle = True # True: loaded wavelets will be repeatedly emitted at each step and copied into internal wavelets array, otherwise only used when interpolation is done and not copied into internal wavelets array
num_wavelet_fields = 1

if (debug):
    print("wavelet_x: (min,max) = ", wavelet_x.min(), wavelet_x.max())
    print("wavelet_y: (min,max) = ", wavelet_y.min(), wavelet_y.max())
    print("wavelet_field: (min,max) = ", wavelet_field.min(), wavelet_field.max())

    print("------ first 10 wavelets ------ ")
    print("wavelet_x: ", wavelet_x[:10])
    print("wavelet_y: ", wavelet_y[:10])
    print("wavelet_field: ", wavelet_field[:10])

####################### Diagnostics ##########################
print_interval = 1        # interval for printing simulation steps
mesh_output_interval = num_step - 1
wavelet_output_interval = num_step - 1

if (mpi_rank==0):
  mesh_output = True
  make_output_dirs(run_name+"/mesh", num_step, mesh_output_start, mesh_output_interval)
  wavelet_output = True 
  make_output_dirs(run_name+"/wavelet", num_step, wavelet_output_start, wavelet_output_interval)
  make_output_dirs(run_name+"/traj", num_step, wavelet_output_start, wavelet_output_interval)
