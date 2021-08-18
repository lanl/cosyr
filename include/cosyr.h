#pragma once

//========================================================================================
// COSYR beam dynamic code
// Copyright(C) 
// Licensed under BSD-3 License, see LICENSE file for details
//========================================================================================
//! \file cosyr.hpp
//  \brief contains cosyr general purpose types, structures, enums, etc.

// kokkos
#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp> 

/* REMARK:
 * no need to define execution and memory space
 * 1. if kokkos is compiled with CUDA support then
 * the default device execution space is Kokkos::Cuda
 * if kokkos is compiled with OpenMP support then
 * it is Kokkos::OpenMP otherwise it's serial.
 * 2. if kokkos is compiled with CudaUVM enabled then
 * the default memory space for views is UVM, otherwise
 * it will be the device memory space which can be
 * the GPU embedded memory if present or the RAM.
 */

// pusher
#include <Cabana_Types.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_Parallel.hpp>

// remap
#include "portage/driver/driver_swarm.h"
#include "portage/search/search_points_bins.h"
#include "portage/search/search_points_by_cells.h"
#include "portage/accumulate/accumulate.h"
#include "portage/estimate/estimate.h"

// kernel
// choice for kernel, default is mixed transverse-field/longitudinal-potential kernel 
#define MIX_KERNEL

// simulation
#ifndef DIM
#define DIM 2
#endif

namespace cosyr {

#define COSYR_VER "0.1.1"

// physics constants in cgs unit
#define ELECTRON_CHARGE 4.80320427e-10     // Fr
#define ELECTRON_MASS   9.10938370e-28     // g
#define LIGHT_SPEED     2.99792458e10      // cm  
#define NUM_ELECTRON_nC (1e-9/1.602176634e-19)

// define parallel policies
using Host        = Kokkos::DefaultHostExecutionSpace;
using HostSpace   = Kokkos::HostSpace;
using HostView    = Kokkos::View<double*, HostSpace>;
using HostRange   = Kokkos::RangePolicy<Host>;
using Mirror      = Kokkos::View<double*>::HostMirror;
using MirrorPoint = Kokkos::View<double*[DIM]>::HostMirror;
using MirrorField = Kokkos::vector<Kokkos::View<double*>::HostMirror>;
using View        = Kokkos::View<double*>;
using ViewPoint   = Kokkos::View<double*[DIM]>; // TODO: switch to Kokkos::View<Wonton::Point<dim>*>
using ViewField   = Kokkos::vector<Kokkos::View<double*>>;
using Remapper    = Portage::SwarmDriver<Portage::SearchPointsBins,
                                         Portage::Meshfree::Accumulate,
                                         Portage::Meshfree::Estimate,
                                         DIM, Wonton::Swarm<DIM>,
                                         Wonton::SwarmState<DIM>>;
}

#ifdef ENABLE_THRUST
  #include "thrust/device_vector.h"
  #include "thrust/transform.h"
#endif

// mpi
#include <mpi.h>

// utils
#include "timer.h"

/**
 * @brief
 *
 * - horizontal in bending plane.
 * - vertical in bending plane.
 * - normal to bending plane.
 */
#if DIM == 3
  enum Coord { X, Y, Z };
#else
  enum Coord { X, Y };
#endif

/**
 * @brief
 */
#if DIM==3
  enum Field { F1, F2, F3, F4 };
#else
  enum Field { F1, F2, F3 };
#endif

// Particle trajectory quantities and their storage order in host view 
#define TRAJ_POS_X 0
#define TRAJ_POS_Y 1
#if DIM==3
#define TRAJ_POS_Z 2
#define NUM_TRAJ_QUANTITIES 3
#else
#define NUM_TRAJ_QUANTITIES 2
#endif

// Particle quantities and their storage order in host view 
#define PART_POS_X 0
#define PART_POS_Y 1
#define PART_MOM_X 0
#define PART_MOM_Y 1
#if DIM==3
#define PART_POS_Z 2
#define PART_MOM_Z 2
#define NUM_PARTICLE_QUANTITIES 6
#else
#define NUM_PARTICLE_QUANTITIES 4
#endif

// Wavelet quantities and their storage order in device view 
#define WT_POS_X 0
#define WT_POS_Y 1
#define WT_FLD1  0
#define WT_FLD2  1
#define WT_FLD3  2
#if DIM==3
#define NUM_WAVELET_QUANTITIES 7
#define WT_POS_Z 2
#define WT_FLD4  3
#else
#define NUM_WAVELET_QUANTITIES 5
#endif

// Wavelet 

// Emission quantities and their storage order in device view and host mirror
#define NUM_EMT_QUANTITIES 8
#define EMT_TIME 0
#define EMT_POS_X 1
#define EMT_POS_Y 2
#define EMT_VEL_X 3
#define EMT_VEL_Y 4
#define EMT_ACC_X 5
#define EMT_ACC_Y 6
#define EMT_GAMMA 7

// copy from host to device
#define do_deep_copy true
