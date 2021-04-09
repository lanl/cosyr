# Cosyr


###### Summary
Cosyr is a particle beam dynamics simulation code with multi-dimensional [synchrotron radiation](https://en.wikipedia.org/wiki/Synchrotron_radiation) effects. It tackles a fundamental problem of the nonlinear dynamics of a particle beam from its complete self-fields, particularly the radiation fields, i.e the [coherent synchrotron radiation (CSR)](http://linkinghub.elsevier.com/retrieve/pii/S016890029700822X) problem. The latter underpins many accelerator design issues in ultra-bright beam applications, as well as those arising in the development of advanced accelerators. It is written in C++ and supports two levels of parallelism with MPI and Kokkos.

###### Algorithms
Cosyr consists of three components:

- a field/wavelet computation kernel,
- a wavelet-to-mesh interpolation module,
- a particle pusher. 

The particle pusher is similar to those for existing high performance kinetic plasma simulation codes, such as [VPIC](https://github.com/lanl/vpic). Unlike other particle-mesh codes with a local PDE-based field solver that communicates only with neighbors, Cosyr's field solver is based on the retarded Green's function and thus nonlocal both in time and space. Such an approach allows both the decoupling of the time/spatial scales in coherent and incoherent effects, and the improved solution to the beam self-fields. 

###### Performance

Cosyr is designed to leverage multi-GPU nodes. It exploits the fact that the field kernel, wavelets emission and particle update are completely local. For a given beam, a subset of particles is assigned to a MPI rank. The field kernel is primarily parallelized over particles to be run on manycore nodes or GPU. The wavelet-to-mesh interpolation is done in a multithreaded way using [Portage](https://github.com/laristra/portage).

### Build and use

It can be built on Linux or macOS using [CMake](https://cmake.org).  
It requires a [C++14](https://isocpp.org/wiki/faq/cpp14-language) compiler endowed with [OpenMP](https://www.openmp.org).  
The field kernels are designed to be run on manycore CPU or GPU.  
For optimized builds on Intel KNL, we recommend the Intel compiler.  
To run on Nvidia's GPU, it should be compiled with [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).  
> It was not currently tested on AMD's GPU.  

###### Dependencies

- [kokkos](https://github.com/kokkos/kokkos) for the wavefronts and field kernels.
- [cabana](https://github.com/ECP-copa/Cabana) for the particle pusher.
- [portage](https://github.com/laristra/portage) for the interpolation step.
- [pybind](https://github.com/pybind/pybind11) and a [python](https://www.python.org) environment to handle simulation parameters.  

The instructions to build the project and its dependencies can be found [here](./docs/BUILD.md).   

###### Running

The simulation parameters are given through an input deck.  
The latter is simply a regular Python script.  
Those parameters are loaded into the Python locals and can changed as needed.  
A set of examples are given in the [input](./input) directory.  
It is also possible to initialize a beam array for the simulation as shown [here](./input/test_beam_remap.py).  
To run: `mpirun -np 4 ./cosyr params.py`

### License and contributions

Cosyr is developed at Los Alamos National Laboratory (C20129).  
It is supported by the Laboratory Directed Research and Development program (LDRD).  
It is open source under the [BSD-3 licence](./LICENSE.txt).  
It is developed by:

- Chengkun Huang 
- Feiyu Li 
- Hoby Rakotoarivelo
- Boqian Shen 

Other contributors:

- Rao Garimella 
- Thomas Kwan
- Bruce Carlsten 
- Robert Robey 
- Orion Yeung
- Parker Pombrio

### Copyright
© 2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
