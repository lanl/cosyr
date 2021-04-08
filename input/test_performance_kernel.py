# -----------------------------------------
# -  Input deck for realistic beam size   -
# -----------------------------------------

import numpy as np
from input.utils import *
from input.misc import *

####################### Preprocessing ##########################

run_name = "test_performance_kernel"

gamma=1000
# Ct=2  
delta_alpha_B = 0.2  # assume the bending section spreads over \Delta\alpha=0.2
l_beam = 3000        # um
r_beam = 50          # um
delta_alpha_tilde = 10000 #gamma * delta_alpha_B / 2
delta_chi_tilde = 2 * r_beam * 1e-6 * gamma**2.0

print("delta_alpha_tilde={}, delta_chi_tilde={}".format(delta_alpha_tilde, delta_chi_tilde))

mesh_alpha_npt = 100 #int(delta_alpha_tilde/1.0+1.0)
mesh_chi_npt = 100 #int(delta_chi_tilde/1.0+1.0)

if not mesh_alpha_npt%2:
    mesh_alpha_npt += 1
if not mesh_chi_npt%2:
    mesh_chi_npt += 1

print("mesh_alpha_npt={}, mesh_chi_npt={}".format(mesh_alpha_npt, mesh_chi_npt))

Nw = 2000

####################### Main setup ##########################

# --- kernel mesh info ---
num_wavefronts = 2000              # number of wavefronts
num_dirs = 100                     # number of field lines
num_step = num_wavefronts          # number of steps
dt = delta_alpha_B/num_wavefronts  # time step in electron rest frame
# dt=Ct/(gamma*Nw)
remap_interval = num_step - 1      # interval of doing remapping (in time steps)
remap_scatter = False              # use scatter weights form for remap
remap_scaling[0] = 1.0             # support/smoothing length scaling factor
remap_scaling[1] = 1.0             # support/smoothing length scaling factor

# --- eletron beam info ---
num_particles = 100                  # number of particles
trajectory_type = 2                # 1: straight line, 2: circular, 3: sinusoidal
parameters[0] = gamma              # parameters for describing the motion of electron
parameters[1] = 1

def init_beam(npart, gamma, delta_gamma, emittance):
    import numpy as np
    # [x, y, px, py]
    part = np.zeros([npart,4])
    part[:,1] = 1.0                # set y0=1
    return (part)

beam = init_beam(num_particles, 10.0, 0.0, 0.0)
del beam                           # init a single particle instead

# --- comoving mesh info ---
num_gridpt_hor = mesh_alpha_npt                # number of points in x-axis
num_gridpt_ver = mesh_chi_npt                  # number of points in y-axis
mesh_span_angle = delta_alpha_tilde/gamma**3   # in radians
mesh_width = delta_chi_tilde/gamma**2          # in unit of radius

# --- wavelets ---
def load_wavelets(filename, assign_field_value=True) :
    import numpy as np
    data = np.genfromtxt(filename, delimiter=",") # read CSV file
    #data = dat[:1000,:]
    nwavelets = data.shape[0]
    field = np.zeros(nwavelets)

    def get_field(pos):
        x = pos[:,0]
        y = pos[:,1]
        ymax = y.max()
        ymin = y.min()
        print("ymin = {}, ymax = {}".format(ymin, ymax))
        return np.cos(5.0*y*2*np.pi/(ymax-ymin))   # test case, assign test field of cos(x)

    if (assign_field_value) : 
        field = get_field(data[:,:2])

    # for val in field:
    #     print("exact_value = {}".format(val))

    # ensure we pass different array addresses for wavelet x,y coordinate 
    wx = data[:,0].copy()
    wy = data[:,1].copy()
    return wx, wy, field

path = "../input/wavelets/analytic_test/wavelets_adaptative.csv"
#wavelet_x, wavelet_y, wavelet_field = load_wavelets(path)
## unscale if necessary
#wavelet_x /= gamma**3.0
#wavelet_y /= gamma**2.0
#wavelet_field *= gamma**4.0
del wavelet_x, wavelet_y, wavelet_field
min_emit_angle = 10.0
wavelet_type = 1   # 0: use global (x,y) coordinate; 1: use local (x',y') coordinate; 2: (TODO) use local cylindrical coordinate
use_wavelet_for_subcycle = False # True: loaded wavelets will be repeatedly emitted at each step and copied into internal wavelets array, otherwise only used when interpolation is done and not copied into internal wavelets array
num_wavelet_fields = 1

if (wavelet_type == 0) :
    # convert to global (x,y) coordinate
    wavelet_x_prime = wavelet_x.copy()
    wavelet_y_prime = wavelet_y + 1.0
    rotation_angle = 0.19989890421779444  # positive for counter clock-wise rotation
    wavelet_x = wavelet_x_prime*np.cos(rotation_angle) + wavelet_y_prime*np.sin(rotation_angle)
    wavelet_y = -wavelet_x_prime*np.sin(rotation_angle) + wavelet_y_prime*np.cos(rotation_angle)

# post-process options
postprocess = False
field_func = "cos(5.*y*6.28318530718/100)"
field_scale = (gamma**4.0)
coord_scale[0] = (gamma**3.0)
coord_scale[1] = (gamma**2.0)
coord_shift[0] = 0.0
coord_shift[1] = 0.0
file_exact = "data/exact_values.csv"
file_error = "data/error_map.csv"

####################### Diagnostics ##########################
print_interval = 100        # interval for printing simulation steps
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