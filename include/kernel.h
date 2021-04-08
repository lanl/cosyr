#pragma once

#include "cosyr.h"
#include "beam.h"

/**
 * @brief Kernels for field calculations.
 *
 */
namespace cosyr { namespace kernel {

/**
 * @brief Emit wavefronts of given particle at current time.
 *
 * @param emit_info: wavefront emission quantities.
 * @param emit_current: current emitted wavefront quantities.
 * @param num_active: number of active wavelets.
 * @param loaded_wavelet: loaded wavelets.
 * @param loaded_field: loaded wavelet radiation fields.
 * @param emitted_wavelet: emitted wavelets.
 * @param emitted_field: emitted wavelet radiation fields.
 * @param num_loaded_wavelets: number of loaded wavelets.
 * @param num_wavefronts: number of wavefronts.
 * @param num_dirs: number of field directions.
 * @param step: current step.
 * @param j: current particle index.
 * @param cos_mc_angle: cosine of mesh center angle.
 * @param sin_mc_angle: sinus of mesh center angle.
 * @param mesh_radius: radius of radial mesh.
 * @param use_subcycle_wavelets: use loaded wavelets for subcycling.
 */
KOKKOS_FUNCTION
void emit_wavefronts(Kokkos::View<double*> emit_info,
                     Kokkos::View<double*> emit_current,
                     Kokkos::View<int*> num_active,
                     Kokkos::View<double*[DIM]> loaded_wavelet,
                     Kokkos::vector<Kokkos::View<double*>> loaded_field,
                     Kokkos::View<double*[DIM]> emitted_wavelet,
                     Kokkos::vector<Kokkos::View<double*>> emitted_field,
                     int num_loaded_wavelets,
                     int num_wavefronts,
                     int num_dirs,
                     int step,
                     int j,
                     double cos_mc_angle,
                     double sin_mc_angle,
                     double mesh_radius,
                     bool use_subcycle_wavelets);

/**
 * @brief Find intersection of given wavefront with moving mesh.
 *
 * @param emit_info: wavefront emission quantities.
 * @param index_emit_wave: index of wavefront in emit_info.
 * @param dt_emit: time since emission.
 * @param mesh_halfwidth_x: half width of mesh.
 * @param mesh_halfwidth_y: half height of mesh.
 * @param cos_mc_angle: cosine of mesh center angle.
 * @param sin_mc_angle: sinus of mesh center angle.
 * @param mesh_radius: radius of radial mesh.
 * @param angle_range: angle extents (clockwise) of each range.
 * @param angle_running_sum: angular span.
 * @return angle range count.
 */
KOKKOS_FUNCTION
int intersect_wavefronts_mesh(Kokkos::View<double*> emit_info,
                              int index_emit_wave,
                              double dt_emit,
                              double mesh_halfwidth_x,
                              double mesh_halfwidth_y,
                              double cos_mc_angle,
                              double sin_mc_angle,
                              double mesh_radius,
                              double angle_range[4][2],
                              double* angle_running_sum);

/**
 * @brief Calculate radiation field of a given wavefront.
 *
 * @param emit_info: wavefront emission quantities.
 * @param num_active: number of active wavelets.
 * @param emitted_wavelet: emitted wavelets.
 * @param emitted_field: emitted wavelets radiation field.
 * @param index_particle: index of current particle.
 * @param index_wavelet: index of first wavelet of wavefront.
 * @param index_emit_wave: index of wavefront in emit_info.
 * @param dt_emt: time since emission.
 * @param q: charge per particle.
 * @param min_emit_angle: minimum emission angle.
 * @param num_dirs: number of field directions.
 * @param range_count: angle range count.
 * @param angle_range: angle extents.
 * @param angle_running_sum: angular span.
 * @param mesh_center_x: absissa of mesh center.
 * @param mesh_center_y: ordinate of mesh center.
 * @param mesh_center_angle: angle of mesh center.
 * @param cos_mc_angle: cosine of mesh center.
 * @param sin_mc_angle: sinus of mesh center.
 */
KOKKOS_FUNCTION
void calculate_fields(Kokkos::View<double*> emit_info,
                      Kokkos::View<int*> num_active,
                      Kokkos::View<double*[DIM]> emitted_wavelet,
                      Kokkos::vector<Kokkos::View<double*>> emitted_field,
                      int index_particle,
                      int index_wavelet,
                      int index_emit_wave,
                      double dt_emt,
                      double q,
                      double min_emit_angle,
                      int num_dirs,
                      int range_count,
                      double angle_range[4][2],
                      const double* angle_running_sum,
                      double mesh_center_x,
                      double mesh_center_y,
                      double mesh_center_angle,
                      double cos_mc_angle,
                      double sin_mc_angle);

}} // namespace cosyr::kernel