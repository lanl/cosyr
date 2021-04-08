
###### Note on parallel HDF5

0. First of all, your `PYTHONPATH` should be correctly set to the one corresponding to the python intrepeter used in pybind:
 
```export PYTHONPATH="/path/to/python/site-packages"```

It is recommended that you use python virtual environment to install mpi4py and parallel h5py.

1. You will need to install `mpi4py` and `cached_property`


```
pip3 install mpi4py
pip3 install cached-property
```

You can check that it works by testing:

```
python3 hello.py

# hello.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("hello world from process ", rank)
```

2. You will need a parallel variant of hdf5. The one shipped by default by homebrew is not.

```
brew uninstall --ignore-dependencies hdf5
brew install hdf5-mpi
```

You can then check that it works using:

`h5pcc -showconfig`

3. You will need to build h5py manually since the variant installed with pip doesn’t support MPI. Besides, you cannot directly install in system directories so you will have to specify the path.

```bash
cd csr/dependencies
git clone git@github.com:h5py/h5py.git
python3 setup.py configure --mpi —hdf5=/usr/local/Cellar/hdf5-mpi/1.12.0
python3 setup.py install —home=./install
mv ./install/lib/python/h5py-2.10.0-py3.8-macosx-10.15-x86_64.egg/h5py $PYTHONPATH
```

4. You will have to set the correct python interpreter in CMake for pybind11 by adding:

`-DPYTHON_EXECUTABLE=//path/to/python/python3`
