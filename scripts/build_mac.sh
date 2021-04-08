#!/bin/sh

set -e

rm -fr build
mkdir build 
cd build

dependency_root="/Users/huangck/Documents/Work/CSR/2019-ER/ThirdPartyLibraries"
pybind_root="${dependency_root}/pybind11_install"
portage_root="${dependency_root}/Portage_install"
thrust_root="${dependency_root}/Thrust/thrust"
kokkos_root="${dependency_root}/Kokkos_install"

cmake -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-lutil" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_THRUST=True \
  -DPORTAGE_DIR="${portage_root}" \
  -DKokkos_DIR="${kokkos_root}" \
  -DTHRUST_DIR="${thrust_root}" \
  -Dpybind11_DIR="${pybind_root}" \
..

gmake VERBOSE=1
