import numpy as np

class CosyrAnalyze(object):
   
    def __init__(self, gamma, data_dir="../..", # path to data
                 charge=0.01e-9, # beam charge in Coulomb
                 R_bend=1.0,  # beam bending radius in meter
                 pid=0, # particle index
                 step=0, # time step
                 dt=0.0001, # time step size
                 traj_type = 2, # 1: straightline, 2: synchrotron, 3: undulator
                 load_data_all=0, # if to load all data
                 wf_xy_rotate = 0, # rotate wavefront positions and co-moving mesh from x-y to x'-y'
                 wf_xy2polar =0, # convert wavefront positions from x-y to polar coordinates
                 p_beam = None,  # ocelot p_array for beam
                ):

        self = locals().pop("self")
        for name, val in locals().items():
            print("setting " + name + " to ", val)
            setattr(self, name, val)
        
        if self.load_data_all:
            self.load_trajectory()
            self.load_wavefronts()
            self.load_cmesh()
            self.load_beam_csv()
            #self.load_cmesh_polar()
            print("done reading.")
        
        if self.wf_xy_rotate:
            from scipy.spatial.transform import Rotation as R
            if not self.load_data_all:
                self.load_trajectory()
                self.load_wavefronts()
                self.load_cmesh()                
            r0 = np.sqrt(self.curr_x**2+self.curr_y**2)
            alpha0 = np.arctan2(self.curr_x, self.curr_y)
            print("alpha0 = ", alpha0)
            r = R.from_euler('z', alpha0)
            rot_matrix = r.as_matrix()
            print("rot_matrix=", rot_matrix)
            print("curr_prime = ", r.apply(np.asarray([self.curr_x, self.curr_y, 0.0])))
            self.wf_xprime = self.wf_x * rot_matrix[0,0] + self.wf_y * rot_matrix[0,1]
            self.wf_yprime = self.wf_x * rot_matrix[1,0] + self.wf_y * rot_matrix[1,1]
            self.cmesh_xprime = self.cmesh_x * rot_matrix[0,0] + self.cmesh_y * rot_matrix[0,1]
            self.cmesh_yprime = self.cmesh_x * rot_matrix[1,0] + self.cmesh_y * rot_matrix[1,1]            
            # axis range
            self.xprime_lim_cmesh = [self.cmesh_xprime.min(), self.cmesh_xprime.max()] 
            self.yprime_lim_cmesh = [self.cmesh_yprime.min(), self.cmesh_yprime.max()] 
            
        if self.wf_xy2polar:
            if not self.load_data_all:
                self.load_trajectory()
                self.load_wavefronts()
            r0 = np.sqrt(self.curr_x**2+self.curr_y**2)
            alpha0 = np.arctan2(self.curr_y, self.curr_x)
            self.wf_alpha = np.zeros(self.wf_x.shape[0])
            self.wf_chi = np.zeros(self.wf_x.shape[0])
            for i in range(self.wf_x.shape[0]):
                r = np.sqrt(self.wf_x[i]**2+self.wf_y[i]**2)
                alpha = np.arctan2(self.wf_y[i], self.wf_x[i])
                self.wf_alpha[i] = alpha0 - alpha
                self.wf_chi[i] = r - r0


    def load_trajectory(self):
        import pandas as pd

        # electron trajectory
        #self.traj_x, self.traj_y = np.loadtxt(
        #    self.data_dir+'/trajectory_{}_0.csv'.format(self.pid), delimiter=',', unpack=True)
        d = pd.read_csv(self.data_dir+'/traj/{}/trajectory_{}_0.csv'.format(self.step, self.pid), delimiter=",", dtype="float64").values
        self.traj_x = d[:,0]
        self.traj_y = d[:,1]
        # Current electron position
        self.curr_x = 0.5*(self.traj_x[-1] + self.traj_x[-2]) 
        self.curr_y = 0.5*(self.traj_y[-1] + self.traj_y[-2])
        # predefined trajectory according to trajectory type
        self.rotation_angle = 0.0
        if self.traj_type==2:
            self.traj_x_predef = np.linspace(0,1,1000)
            self.traj_y_predef = np.sqrt(1-self.traj_x_predef**2)
        

    def load_wavefronts(self):
        import pandas as pd

        # positions
        #self.wf_x, self.wf_y = np.loadtxt(
        #    self.data_dir+'/wavefronts_{}_0.csv'.format(self.pid), delimiter=',', unpack=True)
        d = pd.read_csv(self.data_dir+'/wavelet/{}/wavefronts_{}_0.csv'.format(self.step, self.pid), delimiter=",", dtype="float64").values
        self.wf_x = d[:,0]
        self.wf_y = d[:,1]
        # axis range
        self.xlim_wf = [self.wf_x.min(), self.wf_x.max()] 
        self.ylim_wf = [self.wf_y.min(), self.wf_y.max()]
        # fields 
        #self.wf_vfld, self.wf_afld, self.wf_tfld = np.loadtxt(
        #    self.data_dir+'/field_{}_0.csv'.format(self.pid), delimiter=',', unpack=True)
        d = pd.read_csv(self.data_dir+'/wavelet/{}/field_{}_0.csv'.format(self.step, self.pid), delimiter=",", dtype=np.float64).values
        self.wf_fld1 = d[:,0]
        self.wf_fld2 = d[:,1]
        self.wf_fld3 = d[:,2]
        self.wf_fld1 /= self.gamma**4.0
        self.wf_fld2 /= self.gamma**4.0
        self.wf_fld3 /= self.gamma**4.0

        
    def save_wavefronts(self, path, l_beam, d_beam, wx, wy, wf1, wf2, wf3, scale_coord=True, unscale_field=True, format="npy", load_type="sub"):
        from os import mkdir
      
        path2subcycling = path +'/g{}-{}x{}um-{}'.format(np.int(self.gamma), l_beam, d_beam, load_type)
        print("saving wavelets into ", path2subcycling)
        try:
            mkdir(path2subcycling)
        except OSError as error:
            print(error)     

        fld_sub = np.zeros([3, wx.shape[0]])
        fld_sub[0,:] = wf1
        fld_sub[1,:] = wf2
        fld_sub[2,:] = wf3

        scaled_xprime_sub = wx.copy()
        scaled_yprime_sub = wy.copy()
        
        if (scale_coord) :
           scaled_xprime_sub *= self.gamma**3.0
           scaled_yprime_sub *= self.gamma**2.0
        
        if (unscale_field) :
           fld_sub *= self.gamma**4.0
        
        # save as csv
        if (format == "csv") :
           np.savetxt(path2subcycling+'/scaled_xprime_{}.csv'.format(load_type[0:3]), scaled_xprime_sub, delimiter=',')
           np.savetxt(path2subcycling+'/scaled_yprime_{}.csv'.format(load_type[0:3]), scaled_yprime_sub, delimiter=',')
           np.savetxt(path2subcycling+'/EBRad_{}.csv'.format(load_type[0:3]), fld_sub, delimiter=',')
        
        # save as npy
        if (format == "npy") :
           np.save(path2subcycling+'/scaled_xprime_{}.npy'.format(load_type[0:3]), scaled_xprime_sub)
           np.save(path2subcycling+'/scaled_yprime_{}.npy'.format(load_type[0:3]), scaled_yprime_sub)
           np.save(path2subcycling+'/EBRad_{}.npy'.format(load_type[0:3]), fld_sub)

               

    def load_cmesh(self):
        from scipy.spatial.transform import Rotation as R
        import pandas as pd
        import os

        # positions
        #self.cmesh_x, self.cmesh_y = np.loadtxt(
        #    self.data_dir+'/comoving_mesh_pos.csv', delimiter=',', unpack=True)
        pos_file = self.data_dir+'/mesh/{}/comoving_mesh_pos.csv'.format(self.step)
        d = pd.read_csv(pos_file, delimiter=",", dtype="float64").values
        self.cmesh_x = d[:,0]
        self.cmesh_y = d[:,1]
        # axis range
        self.xlim_cmesh = [self.cmesh_x.min(), self.cmesh_x.max()] 
        self.ylim_cmesh = [self.cmesh_y.min(), self.cmesh_y.max()]
        
        # fields
        #self.cmesh_vfld, self.cmesh_afld, self.cmesh_tfld = np.loadtxt(
        #    self.data_dir+'/comoving_mesh_field.csv', delimiter=',', unpack=True)
        fld_file = self.data_dir+'/mesh/{}/comoving_mesh_field.csv'.format(self.step)
        d = pd.read_csv(fld_file, delimiter=",", dtype="float64").values
        self.cmesh_fld1 = d[:,0]
        self.cmesh_fld2 = d[:,1]
        self.cmesh_fld3 = d[:,2]        
        self.cmesh_fld1 /= self.gamma**4.0
        self.cmesh_fld2 /= self.gamma**4.0
        self.cmesh_fld3 /= self.gamma**4.0

        # gradient 
        gradient_file = self.data_dir+'/mesh/{}/comoving_mesh_gradients.csv'.format(self.step)
        if (os.path.isfile(gradient_file)) :
            d = pd.read_csv(gradient_file, delimiter=",", dtype="float64").values
            self.cmesh_fld1_dx = d[:,0]
            self.cmesh_fld1_dy = d[:,1]
            self.cmesh_fld2_dx = d[:,2]
            self.cmesh_fld2_dy = d[:,3]
            self.cmesh_fld3_dx = d[:,4]
            self.cmesh_fld3_dy = d[:,5]
            self.cmesh_fld1_dx /= self.gamma**4.0
            self.cmesh_fld1_dy /= self.gamma**4.0
            self.cmesh_fld2_dx /= self.gamma**4.0
            self.cmesh_fld2_dy /= self.gamma**4.0
            self.cmesh_fld3_dx /= self.gamma**4.0
            self.cmesh_fld3_dy /= self.gamma**4.0

        # wavelet distrib
        stencil_file = self.data_dir+'/mesh/{}/comoving_mesh_stencil.csv'.format(self.step)
        if (os.path.isfile(stencil_file)) :
            d = pd.read_csv(stencil_file, delimiter=",", dtype="int").values
            self.cmesh_wavelet_distrib = d[:,0]

        # smoothing lengths
        smoothing_file = self.data_dir+'/mesh/{}/comoving_mesh_smoothing.csv'.format(self.step)
        if (os.path.isfile(smoothing_file)) :
            d = pd.read_csv(smoothing_file, delimiter=",", dtype="float64").values
            print(d)
            self.cmesh_smoothing_x = d[:,0]
            self.cmesh_smoothing_y = d[:,1]


    def load_cmesh_polar(self):

        # positions, fields
#         self.cmesh_polar_ang, self.cmesh_polar_rad, self.cmesh_polar_vfld, \
#             self.cmesh_polar_afld, self.cmesh_polar_tfld = np.loadtxt(
#                             self.data_dir+'/data/comoving_mesh_rad_ang.data', delimiter=',', unpack=True)
        self.cmesh_polar_ang, self.cmesh_polar_rad, self.cmesh_polar_afld, =\
            np.loadtxt(self.data_dir+'/comoving_mesh_rad_ang.csv', delimiter=',', unpack=True)
        self.cmesh_polar_rad -= 1.0 # change r to chi=r-1
        # axis range
        self.xlim_cmesh_ang = [self.cmesh_polar_ang.min(), self.cmesh_polar_ang.max()] 
        self.ylim_cmesh_rad = [self.cmesh_polar_rad.min(), self.cmesh_polar_rad.max()]
        self.cmesh_polar_afld /= self.gamma**4.0
        

    def load_beam_csv(self, convert2local=True):
        import fnmatch
        import os
        import pandas as pd
        # use dask for fast csv read
        #https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e
        from dask import dataframe as dd
        import dask.multiprocessing
        import progressbar

        particlefile_path = self.data_dir + "/beam/" + str(self.step) + "/"
        print(particlefile_path)
        self.beam = None
        file_list = os.listdir(particlefile_path)
        bar = progressbar.ProgressBar(max_value=len(file_list))
        ifile = 0
        for file in file_list:
            #print(file)
            if fnmatch.fnmatch(file, 'particles*.csv'):
               #beam_temp=np.genfromtxt(path + file,delimiter=",")
               #_beam_buffer = pd.read_csv(particlefile_path + file, delimiter=",", dtype="float64").values
               _beam_buffer = dd.read_csv(particlefile_path + file, 
                                   sep=",", dtype="float64",
                                   #blocksize=16 * 1024 * 1024, # 16MB chunks)
                                   ).values.compute()
               if (self.beam is None) : 
                  self.beam = _beam_buffer 
               else :
                  self.beam = np.append(self.beam, _beam_buffer[1:,:], axis=0)
            ifile += 1
            bar.update(ifile)

        # change to local coordinates
        if (convert2local) :
           print("converting to local coordinate...")
           x0=self.beam[0,0]
           y0=self.beam[0,1] 
           theta = np.arctan2(y0,x0)
           print("beam rotation angle =", np.pi/2.0-theta)
           x_prime = (self.beam[:,0] - x0) * np.sin(theta) - (self.beam[:,1] - y0) * np.cos(theta)
           y_prime = (self.beam[:,0] - x0) * np.cos(theta) + (self.beam[:,1] - y0) * np.sin(theta)
           py_prime = self.beam[:,2] * np.cos(theta) + self.beam[:,3] * np.sin(theta)
           px_prime = self.beam[:,2] * np.sin(theta) - self.beam[:,3] * np.cos(theta)
           self.beam[:,0] = x_prime
           self.beam[:,1] = y_prime
           self.beam[:,2] = px_prime
           self.beam[:,3] = py_prime

        print(self.beam.shape[0], " beam particles loaded")
        
        
    def load_beam_hdf5(self) :
        import h5py
        
        h5file = self.data_dir+ "/beam/" + "beam_" + str(self.step) + ".h5"
        hf= h5py.File(h5file, "r")
        self.beam = np.array(hf.get("beam"))
        hf.close()
        print(self.beam.shape[0], " beam particles loaded")    
        
        
    def save_beam_hdf5(self) :
        import h5py
        
        h5file = self.data_dir+ "/beam/" + "beam_" + str(self.step) + ".h5"
        hf= h5py.File(h5file, "w")
        hf.create_dataset("beam", data=self.beam)
        hf.close()
  

    def show_beam(self, np_slice=300, smooth_param=0.0, figsize=(12,8), tau_unit="um") :
        from ocelot.gui.accelerator import show_e_beam
        import matplotlib.pyplot as plt

        if (self.beam is None) : 
            print("no beam loaded")
            return
        elif (self.p_beam is None) : 
            print("converting beam to ocelot p_array ...")
            self.p_beam = self.convert2ocelot(self.beam, self.charge, self.gamma, self.R_bend)
            print("done.")

        show_e_beam(p_array=self.p_beam, nparts_in_slice=np_slice, smooth_param = smooth_param, title="t="+str(np.round(self.step*self.dt,4)), inverse_tau=True, figsize=figsize, tau_units=tau_unit)
        plt.tight_layout(pad=0.6)

    
    # convert beam to ocelot ParticleArray
    def convert2ocelot(self, _beam, _charge, _gamma, _R):
        import ocelot

        npart = _beam.shape[0] - 1
        ref_x0= _beam[0,0]
        ref_y0= _beam[0,1]
        p_array = ocelot.ParticleArray(npart)
        p_array.s = -ref_x0
        p_array.E = _gamma*0.511e-3 # GeV
        p_array.rparticles[0] = (_beam[1:, 1] - ref_y0) * _R        # x, horizontal, in meter
        p_array.rparticles[1] = _beam[1:, 3] / _beam[1:,4]            # x'
        p_array.rparticles[2] = np.zeros_like(_beam[1:, 0])         # y, vertical
        p_array.rparticles[3] = np.zeros_like(_beam[1:, 2]/_gamma)   # y'
        # test data for emittance calculation
        #p_array.rparticles[2] = 1.0e-6*np.random.randn(npart)         # y, vertical
        #p_array.rparticles[3] = 0.01*np.random.randn(npart)         # y'
        p_array.rparticles[4] = -(_beam[1:, 0] - ref_x0) * _R              # tau
        p_array.rparticles[5] = _beam[1:, 4] / _gamma - 1.0         # delta_E/p_0*c
        #p_array.rparticles[5] = np.sqrt(_beam[:, 4]**2.0 - 1.0) / np.sqrt(_gamma**2.0 - 1.0) - 1.0         # delta_p/p_0
        p_array.q_array = _charge/npart* np.ones_like(_beam[1:,0])
        return p_array
