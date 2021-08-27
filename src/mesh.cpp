#include "cosyr.h"
#include <iomanip>
#include <fstream>
#include <vector>
#include "mesh.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
Mesh::Mesh(const Input& input) : input(input) {

  resolution[0] = input.mesh.num_hor;
  resolution[1] = input.mesh.num_ver;
  dimension[0]  = input.mesh.span_angle * input.kernel.radius;
  dimension[1]  = input.mesh.width;
  #if DIM == 3
    resolution[2] = input.mesh.num_normal;
    dimension[2]  = input.mesh.height;
  #endif

  num_points = 1;
  for (int d = 0; d < DIM; ++d) {
    half_width[d] = 0.5 * dimension[d];
    h[d] = dimension[d] / (resolution[d] - 1);
    num_points *= resolution[d];
  }

  points.resize(num_points);
  fields.resize(num_points);
  for (auto& gradient : gradients) { gradient.resize(num_points); }

// clear mesh fields
  auto const range = HostRange(0, num_points);

  for (int f = 0; f < num_fields; ++f) {
    auto slice = get_field_slice(fields, f);
    Kokkos::parallel_for(range, [&](int j){ slice(j) = 0.0; });
  }

  // clear gradients
  for (auto& gradient : gradients) {
    auto dx = Cabana::slice<X>(gradient);
    auto dy = Cabana::slice<Y>(gradient);
    Kokkos::parallel_for(range, [&](int j){ dx(j) = 0.0; });
    Kokkos::parallel_for(range, [&](int j){ dy(j) = 0.0; });
  }
}

/* -------------------------------------------------------------------------- */
void Mesh::move(const double* new_position, const double* new_velocity) {

  auto const frame = input.wavelets.frame;
  center.distance = 0.0;
  center.speed = 0.0;

  for (int d = 0; d < DIM; d++) {
    center.position[d] = new_position[d];
    center.velocity[d] = new_velocity[d];
    center.distance += center.position[d] * center.position[d];
    center.speed += center.velocity[d] * center.velocity[d];
  }

  center.distance = std::sqrt(center.distance);
  center.angle[0] = std::atan2(center.position[0], center.position[1]);
  center.speed = std::sqrt(center.speed); // proper speed
  center.direction[0] = std::atan2(center.velocity[0], center.velocity[1]);

#if DIM == 3
  center.angle[1] = std::acos(center.position[2] / center.distance);
  center.direction[1] = std::acos(center.velocity[2] / center.speed);
#endif

  for (int d = 0; d < DIM - 1; ++d) {
    center.sinus_angle[d] = std::sin(center.angle[d]);
    center.cosin_angle[d] = std::cos(center.angle[d]);
  }

  if (frame == LOCAL_CARTESIAN and is_updated) { return; } // no need to update a local mesh

  // TODO: extend to 3D
  switch (frame) {
    case LOCAL_CARTESIAN: {
      // locate the grid points from topleft vertically
      // and rotate the topleft grid point
      auto x = Cabana::slice<X>(points);
      auto y = Cabana::slice<Y>(points);
      double const delta_x = h[0];
      double const delta_y = h[1];

      int const nx = resolution[0];
      int const ny = resolution[1];
      int const nz = 1;

      double const ctr_ix = 0.5 * (nx - 1);
      double const ctr_iy = 0.5 * (ny - 1);
      #if DIM==3
        auto z = Cabana::slice<Z>(mesh_points);
        double delta_z = cell_size[2];
        nz = num_grids[2];
      #endif
      int const count = points.arraySize(0); // number of points in AOSOA

      auto update_points = KOKKOS_LAMBDA(int s, int a) {
        // 1D index of mesh points from SIMD data layout,
        // ip = iz + nz * (iy + ny * ix)
        int index = s * count + a;
        // convert to 2D/3D array index
        int iz = index;
        int iy = iz / nz;
        int ix = iy / ny;
        iy -= ix * ny;
        x.access(s, a) = (ix - ctr_ix) * delta_x;
        y.access(s, a) = (iy - ctr_iy) * delta_y;
        #if DIM==3
          iz -= iy * nz;
          z.access(s,a) = iz * delta_z;
        #endif
      };

      Cabana::SimdPolicy<16, Host> simd_policy(0, num_points);
      Cabana::simd_parallel_for(simd_policy, update_points, "update_points" );
      break;
    }

#ifdef FIXED_LOCAL_CYLNDRIC
    case LOCAL_CYLINDRIC: // (alpha, chi)
      // NOTE: this algorithm works by determining the outer mesh point for an angle, 
      // move radially inward to determine each inner mesh point, then repeat for the next angle. 
      // locate the topleft grid point from the center point of mesh
      double top_x = (radius + half_width) * cos(center_ang);
      double top_y = (radius + half_width) * sin(center_ang);
      double topleft_x = top_x * cos(half_angle) - top_y * sin(half_angle);
      double topleft_y = top_x * sin(half_angle) + top_y * cos(half_angle);

      // locate the grid points from topleft vertically and rotate the topleft grid point
      double deltax;
      double deltay;
      // cannot be parallelized due to iteration dependencies
      for (int i = 0; i < num_grid_pt_hor; i++) { // from small angle to large angle
        int ip = i * num_grid_pt_ver;
        // update deltax and deltay for current radial direction
        deltax = unit_width * topleft_x / (radius + half_width);
        deltay = unit_width * topleft_y / (radius + half_width);    
        pos_hor[ip] = topleft_x - new_center[0]; // top mesh point for each angle, account for the center of local/global coordinate
        pos_ver[ip] = topleft_y - new_center[1];
        // this inner loop move radially inward to calculate each point for a specific angle
        for (int j = 1; j < num_grid_pt_ver; j++) {
          ip += 1;
          pos_hor[ip] = pos_hor[ip - 1] - deltax; // from top to bottom
          pos_ver[ip] = pos_ver[ip - 1] - deltay;
        }// end of j
        top_x = topleft_x;
        top_y = topleft_y;
        topleft_x = top_x * cos_unit_angle + top_y * sin_unit_angle; // rotation to calculate top mesh point for next angle
        topleft_y = -top_x * sin_unit_angle + top_y * cos_unit_angle;
      } // end of i
      break;
#endif
    case GLOBAL_CARTESIAN: {
      center.angle[0] = std::atan2(new_position[0], new_position[1]);
      //TODO: calculate second angle in 3D
      break;
    }

    default: break;
  }

  for (int d = 0; d < DIM - 1; ++d) {
    center.sinus_angle[d] = std::sin(center.angle[d]);
    center.cosin_angle[d] = std::cos(center.angle[d]);
  }

  is_updated = true;
}

/* -------------------------------------------------------------------------- */
void Mesh::sync() {

  if (input.mpi.num_ranks > 1) {
    if (input.mpi.rank == 0) {
      std::cout << "synchronize mesh ... " << std::flush;
    }

    // TODO: use either custom MPI data type or Cabana MPI functionality
    // example: https://github.com/ECP-copa/Cabana/blob/master/example/tutorial
    // /04_aosoa_advanced_unmanaged/advanced_aosoa_unmanaged.cpp

    int const count = num_points * num_fields;
    double buffer[count];
    auto field_slices = { Cabana::slice<F1>(fields),
                          Cabana::slice<F2>(fields),
                          Cabana::slice<F3>(fields) };

    for (int i = 0; i < num_fields; i++) {
      int const offset = i * num_points;
      auto field = field_slices.begin()[i];
      for (int j = 0; j < field.size(); ++j) {
        buffer[j + offset] = field(j);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_SUM, input.mpi.comm);

    for (int i = 0; i < num_fields; i++) {
      int const offset = i * num_points;
      auto field = field_slices.begin()[i];
      for (int j = 0; j < field.size(); ++j) {
        field(j) = buffer[j + offset];
      }
    }

    MPI_Barrier(input.mpi.comm);
    if (input.mpi.rank == 0) { std::cout << "done." << std::endl; }
  }
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr
