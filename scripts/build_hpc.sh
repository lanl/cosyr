#!/bin/bash

# exit if a command fails
set -e

# user options
use_gpu=false
cosyr_root="$(pwd)/.."
supported_clusters=( "kodiak" "darwin" )
supported_compilers=( "gcc" "intel" )
cluster=${supported_clusters[0]}
compiler=${supported_compilers[0]}

print_usage() {
  echo -e "Usage: ./build_hpc.sh [options]"
  echo -e "Options:"
  echo -e "  -h, --help              show this help message and exit"
  echo -e "  -g, --gpu               use GPU as device backend"
  echo -e "  -p, --path <string>     root path of cosyr"
  echo -e "  -c, --cluster <string>  cluster [darwin|kodiak]"
  echo -e "  -l, --compiler <string> compiler [gcc|intel]"
  exit 1
}

parse() {
  # requires GNU getopt
  short_opt="hgpcl"
  long_opt="help,gpu,path:,cluster:,compiler:"
  options=`getopt -o ${short_opt}: -l ${long_opt} -- "$@"`

  # check args
  if [ $? != 0 ]; then
    print_usage
  fi

  # evaluate the option list
  eval set -- "${options}"

  # process args
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help) print_usage;;
      -g|--gpu) use_gpu=true; shift;;
      -p|--path) cosyr_root="$2"; shift;;
      -c|--cluster) cluster="$2"; shift;;
      -l|--compiler) compiler="$2"; shift;;
      --) break;;
      -*) print_usage;;
      *) break;;
    esac
    shift
  done

  if [ ! -f ${cosyr_root}/README.md ]; then
    echo -e "\e[31mError: invalid root path '${cosyr_root}'.\e[0m"
    exit 1
  elif [ "$(echo ${supported_clusters[*]} | grep -c -w ${cluster})" -ne 1 ]; then
    echo -e "\e[31mError: unsupported cluster '${cluster}'\e[0m"
    exit 1
  elif [ "$(echo ${supported_compilers[*]} | grep -c -w ${compiler})" -ne 1 ]; then
    echo -e "\e[31mError: unsupported compiler '${compiler}'\e[0m"
    exit 1
  fi

  # print params
  echo -e "Build options: "
  echo -e "- use GPU: ${use_gpu}"
  echo -e "- root path: ${cosyr_root}"
  echo -e "- cluster: ${cluster}"
  echo -e "- compiler: ${compiler}"
}

build() {
  if [ ${cluster} = "kodiak" ]; then
    # compiler and dependencies versions
    gcc_version=8.3.0
    intel_version=19.0.4
    mpi_version=2.1.2
    cuda_version=10.2
    python_version=3.6
    conda_version=5.0.1
    java_version=1.8.0_45

    # load modules
    module purge
    module load cmake
    if [ ${compiler} = "gcc" ]; then
      module load gcc/${gcc_version}
      module load openmpi/${mpi_version}
    elif [ ${compiler} = "intel" ]; then
      module load intel/${intel_version}
      module load intel-mpi
    fi
    module load java/${java_version}
    module load python/${python_version}-anaconda-${conda_version}
    if ${use_gpu}; then
      module load cudatoolkit/${cuda_version}
    fi
    module list

    # dependencies
    dependency_root="/usr/projects/cosyr/dependencies"

  elif [ ${cluster} = "darwin" ]; then
    # compiler and dependencies versions
    gcc_version=8.2.0
    mpi_version=3.1.3
    cuda_version=10.2
    boost_version=1.70.0

    # load modules
    module purge
    module load cmake/3.17.3
    module load openmpi/${mpi_version}-gcc_${gcc_version}
    if ${use_gpu}; then
      module load cuda/${cuda_version}
      module load boost/${boost_version}
    fi
    module list

    # dependencies
    dependency_root="/projects/cosyr/dependencies"

    # set correct python paths
    export PYTHONPATH="${dependency_root}/python"
    export PATH="${PYTHONPATH}/bin":~/.local/bin:${PATH}
    export LD_LIBRARY_PATH="${PYTHONPATH}/lib":${LD_LIBRARY_PATH}
    export PYTHON_LD_LIBRARY_PATH="${PYTHONPATH}/lib"
  fi

  # set dependencies root locations
  pybind_root="${dependency_root}/pybind"
  boost_root="${dependency_root}/boost"
  portage_root="${dependency_root}/portage"
  kokkos_root="${dependency_root}/kokkos"
  thrust_root="${dependency_root}/thrust"
  tcmalloc_lib="${dependency_root}/gperftools/lib/libtcmalloc.so"

  # purge and create output directory
  cd "${cosyr_root}/build" && rm -rf * && mkdir data

  if ${use_gpu}; then

    [ ${cluster} = "darwin" ] && cuda="cuda-pascal" || cuda="cuda"

    # build
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-lutil" \
      -DCMAKE_CXX_COMPILER=${OMPI_CXX} \
      -DENABLE_THRUST=False \
      -DENABLE_TCMALLOC=True \
      -DPORTAGE_DIR="${portage_root}/boost" \
      -DKokkos_DIR="${kokkos_root}/${cuda}" \
      -DBOOST_ROOT="${boost_root}" \
      -Dpybind11_DIR="${pybind_root}" \
      -DTCMALLOC_LIB="${tcmalloc_lib}" \
      -DPYTHON_EXECUTABLE="$(which python3)" \
      ..
    make -j

    # set compiler, GPU options and thread-core affinity
    export OMPI_CXX=${kokkos_root}/${cuda}/bin/nvcc_wrapper \
           CUDA_VISIBLE_DEVICES=0,1 \
           CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 \
           CUDA_LAUNCH_BLOCKING=1 \
           OMP_NUM_THREADS=18 \
           OMP_PROC_BIND=close \
           OMP_PLACES=cores
  else
    # build
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-lutil" \
      -DENABLE_THRUST=True \
      -DENABLE_TCMALLOC=True \
      -DPORTAGE_DIR="${portage_root}/thrust" \
      -DKokkos_DIR="${kokkos_root}/openmp" \
      -DTHRUST_DIR="${thrust_root}" \
      -Dpybind11_DIR="${pybind_root}" \
      -DTCMALLOC_LIB="${tcmalloc_lib}" \
      -DPYTHON_EXECUTABLE="$(which python3)" \
      ..
    make -j

    # set thread-core affinity
    export OMP_NUM_THREADS=18 \
           OMP_PROC_BIND=close \
           OMP_PLACES=cores
  fi
}

# main
parse "$@" && build