### Build instructions

Below are the instructions to build Cosyr and its dependencies.  
They are assumed to be installed in a directory `dependencies`.

###### Kokkos

```bash
git clone git@github.com:kokkos/kokkos.git
cd kokkos
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/kokkos" \
  -DKokkos_ENABLE_TESTS=Off \
  -DKokkos_ENABLE_SERIAL=On \
  -DKokkos_ENABLE_OPENMP=On \
  -DKokkos_ENABLE_CUDA=[On|Off] \
  -DKokkos_ARCH_KNL=[On|Off] \
  ..
make -j 8
make install
cd ../..
```

###### Cabana

```bash
git clone git@github.com:ECP-copa/Cabana.git cabana
cd cabana
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE="Release" \
  -DCMAKE_PREFIX_PATH="${dependencies}/kokkos" \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/cabana" \
  -DCabana_REQUIRE_OPENMP=ON \
  -DCabana_ENABLE_MPI=ON \
  -DCabana_ENABLE_EXAMPLES=ON \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_PERFORMANCE_TESTING=OFF \
  -DCabana_ENABLE_CAJITA=ON \
  ..
make -j 8
make install
cd ../..
```

###### Pybind

```bash
git clone git@github.com:pybind/pybind11.git pybind
cd pybind
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYTHONPATH}" \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/pybind" \
  -DPYBIND11_TEST=Off \
  ..
make -j 8
make install
cd ../..
```

###### Thrust

```bash
git clone git@github.com:NVIDIA/thrust.git "${dependencies}/thrust"
```

###### Lapacke

```bash
git clone git@github.com:Reference-LAPACK/lapack-release.git lapack
cd lapack
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/lapack" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLAPACKE=ON \
  ..

make -j 8
make install
cd ../..
```

###### Wonton

```bash
git clone --recursive git@github.com:laristra/wonton.git
cd wonton
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/wonton" \
  -DENABLE_UNIT_TESTS=False \
  -DWONTON_ENABLE_MPI=True \
  -DWONTON_ENABLE_THRUST=True \
  -DWONTON_ENABLE_LAPACKE=True \
  -DLAPACKE_ROOT="${dependencies}/lapack" \
  -DTHRUST_ROOT="${dependencies}/thrust" \
  ..

make -j 8
make install
cd ../..
```

###### Portage

```bash
git clone --recursive git@github.com:laristra/portage.git
cd portage
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${dependencies}/portage" \
  -DENABLE_UNIT_TESTS=False \
  -DPORTAGE_ENABLE_MPI=True \
  -DPORTAGE_ENABLE_Jali=False \
  -DPORTAGE_ENABLE_FleCSI=False \
  -DPORTAGE_ENABLE_THRUST=True \
  -DWONTON_ROOT="${dependencies}/wonton" \
  ..

make -j 8
make install
cd ../..
```

###### Cosyr

```bash
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DPORTAGE_DIR="${dependencies}/portage" \
  -DKokkos_DIR="${dependencies}/kokkos" \
  -Dpybind11_DIR="${dependencies}/pybind" \
  -DCabana_DIR="${dependencies}/cabana" \
  ..
make -j 8  
```

A set of Jupyter [notebooks](../analysis) are available for post-processing.

--------------------------------
#### Notes

###### Python

Cosyr relies on scripts to handle simulation parameters and data output.  
For that purpose, a python environment is required.  
It is recommended to install `mpi4py` for faster subcycle wavelets loading.  
To enable HDF5 output, the following packages are required:

- [hdf5](https://www.hdfgroup.org/downloads/hdf5) library (see additional instructions [here](./HDF5.md))
- [h5py](https://github.com/h5py/h5py) to use it through the python interface.
- [mpi4py](https://bitbucket.org/mpi4py/mpi4py/src/master/) to enable parallel output.


###### Incompatibility

It is not currently possible to run both the fields kernels using CUDA backend and the interpolation using OpenMP. There is a [known bug](https://github.com/thrust/thrust/issues/962) in Thrust making it impossible to use other device backends when compiled with [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). 

