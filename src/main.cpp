//! \file main.cpp

#include <cstdlib>

#include "cosyr.h"
#include "input.h"
#include "beam.h"
#include "mesh.h"
#include "remap.h"
#include "kernel.h"
#include "pusher.h"
#include "io.h"
#include "analysis.h"

/**
 * @brief Run the simulation.
 *
 * @param argc: number of arguments.
 * @param argv: arguments values.
 * @return status code.
 */
int main(int argc, char* argv[]) {

  // keep track of time spent on each step
  cosyr::Timer timer;
  // initialize interpreter and keep it alive
  py::scoped_interpreter guard{};
  // parse simulation params and initialize runtime
  cosyr::Input input(argc, argv, timer);

  {
    cosyr::Analysis analysis(input);
    cosyr::Beam beam(input);
    cosyr::Wavelets wavelets(input, timer);
    cosyr::Mesh mesh(input);
    cosyr::Pusher pusher(input, beam, mesh, timer);
    cosyr::Remap remap(input, beam, wavelets, mesh, analysis, timer);
    cosyr::IO io(input, beam, wavelets, mesh, timer);

    double t = 0.;
    double half_dt = input.kernel.dt * 0.5;
    double const q = std::copysign(1.0, beam.q);

    for (int i = 0; i < input.kernel.num_step; i++) {

      input.print_step(i, t);
      t = t + half_dt;
      pusher.move(i, t);
      io.dump_beam(i);

      if (pusher.skip_emission(i)) {
        t = t + half_dt;
        continue;
      }

      if (remap.process(i)) {
        timer.start("kernel");

        // note: does not include the current emission
        int const num_active_wavefront = pusher.num_active_emission;

        // field calculation for wavelet emitted at t=(i+1/2)*dt and mesh at t=(i+1)*dt
        // sin/cos limits set by the four mesh boundaries
        Kokkos::parallel_for(input.kernel.num_particles, KOKKOS_LAMBDA(int index_particle) {

          cosyr::kernel::emit_wavefronts(beam.emit_info,
                                        beam.emit_current,
                                        wavelets.active,
                                        wavelets.loaded.device.coords,
                                        wavelets.loaded.device.fields,
                                        wavelets.emitted.device.coords,
                                        wavelets.emitted.device.fields,
                                        input.wavelets.count,
                                        input.kernel.num_wavefronts,
                                        input.kernel.num_dirs,
                                        num_active_wavefront, index_particle,
                                        mesh.center.cosin_angle[0],
                                        mesh.center.sinus_angle[0],
                                        input.kernel.radius,
                                        input.wavelets.found and input.wavelets.subcycle);

          // loop over wavefronts from most recent one (not current one) to oldest one
          for (int iw = num_active_wavefront-1; iw >= 0; iw--) {

            int const index_wavefront = index_particle * input.kernel.num_wavefronts;
            int const index_emit_wave = (index_wavefront + iw) * NUM_EMT_QUANTITIES;
            int const index_wavelet   = (index_wavefront + num_active_wavefront - iw) * input.kernel.num_dirs;

            double const dt_emit = t - beam.emit_info(index_emit_wave + EMT_TIME); // time since emission

            // angle start and end (clock-wise) of each range.
            double angle_range[4][2] = {{M_PI, -M_PI},
                                        {M_PI, -M_PI},
                                        {M_PI, -M_PI},
                                        {M_PI, -M_PI}};

            double angle_running_sum[5] = {0, 0, 0, 0, 0};

            int range_count = cosyr::kernel::intersect_wavefronts_mesh(beam.emit_info,
                                                                      index_emit_wave,
                                                                      dt_emit,
                                                                      mesh.half_width[0],
                                                                      mesh.half_width[1],
                                                                      mesh.center.cosin_angle[0],
                                                                      mesh.center.sinus_angle[0],
                                                                      input.kernel.radius,
                                                                      angle_range,
                                                                      angle_running_sum);

            cosyr::kernel::calculate_fields(beam.emit_info,
                                            wavelets.active,
                                            wavelets.emitted.device.coords,
                                            wavelets.emitted.device.fields,
                                            index_particle,
                                            index_wavelet,
                                            index_emit_wave,
                                            dt_emit, q,
                                            input.kernel.min_emit_angle,
                                            input.kernel.num_dirs,
                                            range_count,
                                            angle_range,
                                            angle_running_sum,
                                            mesh.center.position[0],
                                            mesh.center.position[1],
                                            mesh.center.angle[0],
                                            mesh.center.cosin_angle[0],
                                            mesh.center.sinus_angle[0]);
          } // end of iw 
        });

        MPI_Barrier(input.mpi.comm);
        timer.stop("kernel");

        remap.interpolate(i, std::abs(beam.q), true);
      }

      io.dump_wavelets(i);
      io.dump_mesh(i);

      t = t + half_dt;
    } // end each time step i

    timer.reset("COSYR timing", input.mpi.rank);
  }

  return input.finalize();
}
