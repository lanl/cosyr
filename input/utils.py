# -----------------------------------------
# -  Input deck utilities for CoSYR   -
# -----------------------------------------

def init_beam(npart, g, lbeam, dbeam, rank):
    import numpy as np
    np.random.seed(rank)
    sgmx, sgmy = lbeam*1e-6/6.0, dbeam*1e-6/6.0
    mean = [0, 1]
    cov = [[sgmx**2, 0], [0, sgmy**2]]
    beam_dist = np.random.multivariate_normal(mean, cov, npart-1)
    part = np.zeros( (npart, 4))  # 4-components [x, y, px, py]
    part[1::,0] = beam_dist[:,0]
    part[1::,1] = beam_dist[:,1]
    part[0,0] = 0.  # set 1st particle as reference particle
    part[0,1] = 1.0 
    beta = np.sqrt(1.0 - g**(-2.0))
    part[:,2] = g*beta 
    part[:,3] = 0.0 

    return (part)


def init_1D_beam(npart, g, lbeam, dbeam, rank):
    # 1D gaussian beam with uniform transverse distribution
    import numpy as np
    np.random.seed(rank)
    sgmx = lbeam*1e-6/6.0
    mean = 0
    beam_dist = np.random.normal(mean, sgmx, npart-1)
    part = np.zeros( (npart, 4))  # 4-components [x, y, px, py]
    part[1::,0] = beam_dist
    if (dbeam == 0.0) :
        part[1::,1] = 1.0
    else :
        part[1::,1] = 2.0*(dbeam*1e-6)*(np.random.rand(npart-1)-0.5) + 1.0  
    part[0,0] = 0.  # set 1st particle as reference particle
    part[0,1] = 1.0 
    beta = np.sqrt(1.0 - g**(-2.0))
    part[:,2] = g*beta 
    part[:,3] = 0.0 

    return (part)


def generate_microbunches(overall_long_beam_env='gaussian', trans_beam_env="gaussian", _gamma=100.0, _npart=200, 
        _nbunches=5, _sgmx_sub_div=4.0, _lbeam=3000, _dbeam=100, _mpi_rank=1):
    import numpy as np
    beam_dist = np.zeros((_npart, 4))
    lbeam_sub = _lbeam*1e-6/_nbunches  # length of each beamlet
    sgmx = _lbeam*1e-6/6.0  # sigma_x for overall beam
    sgmx_sub = lbeam_sub/_sgmx_sub_div  # sigma_x for each beamlet
                            # tuning sgmx_sub_div would adjust the modulation depth
    sgmy_sub = _dbeam*1e-6/6.0  # sigma_y for each beamlet
    beta = np.sqrt(1.0 - _gamma**(-2.0))
    
    # gap between beamlets
    if _nbunches>1:
        gap = _lbeam*1e-6/(_nbunches-1)*0.9
    elif _nbunches == 1: # no microbunching, 'gap' is not really used.
        gap = _lbeam*1e-6  
    
    npart_sub = np.zeros(_nbunches) # part. num per beamlet
    if overall_long_beam_env == 'uniform':
        npart_sub_each = int(_npart/_nbunches)
        npart_sub += npart_sub_each
    elif overall_long_beam_env == 'gaussian':
        npart_rat = np.zeros(_nbunches) # part. number ratio of each beamlet
        for i in range(_nbunches):  # get ratio
            x_sub = (-(_nbunches-1)/2+i)*gap
            npart_rat[i] = np.exp(-x_sub**2/(2*sgmx**2))
        npart_rat_sum = np.sum(npart_rat)
        for i in range(_nbunches):
            npart_sub[i] = int(npart_rat[i]/npart_rat_sum*_npart)
        # make sure total num. per rank is fixed to be npart
        npart_sub[int((_nbunches-1)/2)] = _npart - (np.sum(npart_sub)-npart_sub[int((_nbunches-1)/2)])
#     print ('part# per beamlet per rank:', npart_sub)    

    idx1, idx2 = 0, 0
    np.random.seed(_mpi_rank)
    for j in range(_nbunches):
        mean = [(-(_nbunches-1)/2+j)*gap, 1]
        npart_bunch = int(npart_sub[j])
        if (trans_beam_env == "gaussian") : 
           cov = [[sgmx_sub**2, 0], [0, sgmy_sub**2]]  # diagonal covariance
           beam_xy = np.random.multivariate_normal(mean, cov, npart_bunch)
        elif (trans_beam_env == "uniform") :
           beam_xy = np.zeros([npart_bunch, 2])
           beam_xy[:,0] = np.random.normal(mean[0], sgmx_sub, npart_bunch)
           if (_dbeam == 0.0) :
              beam_xy[:,1] = 1.0
           else :
              beam_xy[:,1] = 2.0*(_dbeam*1e-6)*(np.random.rand(npart_bunch)-0.5) + 1.0  
        idx2 += int(npart_sub[j])
        beam_dist[idx1:idx2, 0] = beam_xy[:,0]
        beam_dist[idx1:idx2, 1] = beam_xy[:,1]
        idx1 += int(npart_sub[j])
    beam_dist[0,0] = 0.  # set 1st particle as reference particle
    beam_dist[0,1] = 1.0     
    beam_dist[:,2] = _gamma*beta 
    beam_dist[:,3] = 0.0 

    return (beam_dist)


def load_file(path, ext="npy"):

    import numpy as np

    data = None
    if ext==".csv" :
       data = np.genfromtxt(path, delimiter=",") 
    if ext in [".npy", ".npz"] :
       data = np.load(path) 

    return data   


## load wavelets
def load_wavelets(path2wavelets, X_coord_file="scaled_xprime_sub.csv", Y_coord_file="scaled_yprime_sub.csv", 
                  fld_file="EsRad_sub.csv", unscale_coord=False, _gamma=10, broadcast=True) :
    import numpy as np
    import os.path

    if (broadcast == True) :
       from mpi4py import MPI
       mpi_rank = MPI.COMM_WORLD.Get_rank()
       comm = MPI.COMM_WORLD

    wx = None
    wy = None
    field = None

    if (broadcast==False or mpi_rank==0) :
       extension = os.path.splitext(X_coord_file)[1]
       wx = load_file(path2wavelets+'/'+X_coord_file, extension) 
       extension = os.path.splitext(Y_coord_file)[1]
       wy = load_file(path2wavelets+'/'+Y_coord_file, extension) 
       extension = os.path.splitext(fld_file)[1]
       field = load_file(path2wavelets+'/'+fld_file, extension) 
       ## unscale if necessary
       if unscale_coord :
          wx /= _gamma**3.0
          wy /= _gamma**2.0

    if (broadcast == True) :
       wx = comm.bcast(wx, root=0)
       wy = comm.bcast(wy, root=0)
       field = comm.bcast(field, root=0)

    return (wx, wy, field)


def load_wavelets_with_analytic_field(filename, assign_field_value=False, unscale_coord=False, _gamma=10, analytic_func=None, *func_args) :
    import numpy as np
    data = np.genfromtxt(filename, delimiter=",") # read CSV file
    nwavelets = data.shape[0]
    field = np.zeros(nwavelets)

    # ensure we pass different array addresses for wavelet x,y coordinate 
    wx = data[:,0].copy()
    wy = data[:,1].copy()

    if (assign_field_value) : 
        field = analytic_func(wx, wy, *func_args)

    # unscale if necessary
    if unscale_coord :
       wx /= _gamma**3.0
       wy /= _gamma**2.0
    field *= _gamma**4.0

    return wx, wy, field


def convert2global(wavelet_x, wavelet_y, rotation_angle=0.0) :
    # convert to global (x,y) coordinate
    # positive rotation_angle for counter clock-wise rotation
    import numpy as np

    wavelet_x_prime = wavelet_x.copy()
    wavelet_y_prime = wavelet_y + 1.0
    wavelet_x = wavelet_x_prime*np.cos(rotation_angle) + wavelet_y_prime*np.sin(rotation_angle)
    wavelet_y = -wavelet_x_prime*np.sin(rotation_angle) + wavelet_y_prime*np.cos(rotation_angle)

    return wavelet_x, wavelet_y


def make_output_dirs(run_name="data", nstep=1, dump_start=0, dump_interval=1) : 
    # make output directory and subdirectoies for each data dump
    import os
    for i in range(0, int((nstep-dump_start)/dump_interval)+1) :
      dir_name = "./" + run_name + "/" + str(int(i*dump_interval)+dump_start)
      os.makedirs(dir_name, exist_ok=True)


# write beam for each rank, this results in separate files
def write_beam(beam, path, rank) :
    import h5py
    hf = h5py.File(path+'/beam_'+ str(rank) +'.h5', 'w')
    hf.create_dataset('beam', data=beam, dtype='f')
    hf.close()

# write beam from each rank into one file, requires mpi4py and parallel h5py
# it is recommended to use python virtual environment and export PYTHONPATH to the corresponding site-package directory
def parallel_write_beam(beam, npart, path, rank) :
    from mpi4py import MPI
    import h5py
    nprocs = MPI.COMM_WORLD.Get_size()
    hf = h5py.File(path+'/beam.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    dset = hf.create_dataset('beam', (nprocs*npart,4), dtype='f')
    dset[rank*npart:(rank+1)*npart,:] = beam
    hf.close()
