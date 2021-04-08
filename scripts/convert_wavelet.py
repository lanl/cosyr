import h5py as h5
import os
import fnmatch
import numpy as np

def write_h5(dset, dir, filename, compression=False) :
    
    h5file = dir+ "/" + filename 
    hf= h5.File(h5file, "w")
    if (compression==True) : 
       hf.create_dataset("data", data=dset, compression='gzip')
    else :
       hf.create_dataset("data", data=dset)
    hf.close()


path=os.getcwd()
format="npy"


file_list = os.listdir(path)

for file in file_list:
    if fnmatch.fnmatch(file, '*.csv'):

       if (format=="h5") :
          h5file_name = os.path.splitext(file)[0] + ".h5"
          print("Converting " + file + " into " + h5file_name)
          data = np.genfromtxt(file, delimiter=",")
          write_h5(data, path, h5file_name) 

       if (format=="npy") :
          npyfile_name = os.path.splitext(file)[0] + ".npy"
          print("Converting " + file + " into " + npyfile_name)
          data = np.genfromtxt(file, delimiter=",")
          np.save(npyfile_name, data)

    
