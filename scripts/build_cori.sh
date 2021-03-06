#!/bin/env bash

# exit on error
set -e

# ---- set modules and environment variables -----------------------------------
# cross compilation for knl
module swap craype-haswell craype-mic-knl
module load python cmake/3.18.2
module list

# use cray wrappers for compilation
export CC="cc" CXX="CC" FC="ftn"
# avoid link error when linking with kokkos
export CRAYPE_LINK_TYPE=dynamic
# default installation directory
dependencies="${CFS}/m3757/dependencies"

# ---- pybind ------------------------------------------------------------------
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

# ---- kokkos ------------------------------------------------------------------
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
  -DKokkos_ENABLE_CUDA=Off \
  -DKokkos_ARCH_KNL=On \
  ..
make -j 8
make install
cd ../..

# ---- cabana ------------------------------------------------------------------
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

# ---- lapacke -----------------------------------------------------------------
git clone git@github.com:Reference-LAPACK/lapack-release.git lapack
cd lapack
# need manual tweak for intel compiler
# see: https://github.com/Reference-LAPACK/lapack/issues/228
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

# ---- thrust ------------------------------------------------------------------
git clone git@github.com:NVIDIA/thrust.git "${dependencies}/thrust"

# ---- wonton ------------------------------------------------------------------
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

# ---- portage -----------------------------------------------------------------
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

# ---- cosyr -------------------------------------------------------------------
git clone "${CFS}/m3757/cosyr.git"
mkdir build
cd build

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TCMALLOC=False \
  -DPORTAGE_DIR="${dependencies}/portage" \
  -DKokkos_DIR="${dependencies}/kokkos" \
  -Dpybind11_DIR="${dependencies}/pybind" \
  -DCabana_DIR="${dependencies}/cabana" \
  ..
make -j 8
