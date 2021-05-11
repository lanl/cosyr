#include "pusher.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
Pusher::Pusher(Input& in_input, Beam& in_beam,
               Mesh& in_mesh, Timer& in_timer)
  : input(in_input),
    beam(in_beam),
    mesh(in_mesh),
    timer(in_timer),
    num_active_emission(0)
{}

/* -------------------------------------------------------------------------- */
void Pusher::move(int step, double t) {

  timer.start("push_particles");

  // update reference particle to x_{1} and v_{1},
  // mesh to x_{1}, and record emission at x_{1/2}
  auto dt = input.kernel.dt; // time step normalized by characteristic frequency
  auto trajectory = input.kernel.trajectory;
  auto motion_params = input.kernel.motion_params;

  beam.move_reference(0, mesh, trajectory, motion_params, step, t, dt);
  mesh.move(beam.reference.coords, beam.reference.momentum);
  beam.move_others(mesh, motion_params, step, dt, t);

  MPI_Barrier(input.mpi.comm);
  timer.stop("push_particles");

  //update_emission_info();
}

/* -------------------------------------------------------------------------- */
void Pusher::update_emission_info() {

  timer.start("emt_info_to_device");

  if (do_deep_copy) {
    // deep copy of new emission event info from host mirror to device
    timer.start("deep_copy_emt_current");
    Kokkos::deep_copy(beam.emit_current, beam.host_emit_current);
    MPI_Barrier(input.mpi.comm);
    timer.stop("deep_copy_emt_current");
  }

  timer.start("reorder_emt_info");

  // reorder current emission info and fill into kk_emt
  Kokkos::parallel_for(input.kernel.num_particles, KOKKOS_LAMBDA(int j) {

    int const index_wavefront = num_active_emission + j * input.kernel.num_wavefronts;
    int const new_index = index_wavefront * NUM_EMT_QUANTITIES;
    int const old_index = j * NUM_EMT_QUANTITIES;

    beam.emit_info[new_index + EMT_TIME ] = beam.emit_current[old_index + EMT_TIME];
    beam.emit_info[new_index + EMT_POS_X] = beam.emit_current[old_index + EMT_POS_X];
    beam.emit_info[new_index + EMT_POS_Y] = beam.emit_current[old_index + EMT_POS_Y];
    beam.emit_info[new_index + EMT_VEL_X] = beam.emit_current[old_index + EMT_VEL_X];
    beam.emit_info[new_index + EMT_VEL_Y] = beam.emit_current[old_index + EMT_VEL_Y];
    beam.emit_info[new_index + EMT_ACC_X] = beam.emit_current[old_index + EMT_ACC_X];
    beam.emit_info[new_index + EMT_ACC_Y] = beam.emit_current[old_index + EMT_ACC_Y];
    beam.emit_info[new_index + EMT_GAMMA] = beam.emit_current[old_index + EMT_GAMMA];

//    if (step == 98) {
//      std::cout << "beam.current["<< old_index <<"]: x = "
//                << beam.emit_current[old_index + EMT_POS_X]
//                << ", y = " << beam.emit_current[old_index + EMT_POS_X] << ")" << std::endl;
//    }
  });

  MPI_Barrier(input.mpi.comm);
  timer.stop("reorder_emt_info");
  timer.stop("emt_info_to_device");
}

/* -------------------------------------------------------------------------- */
bool Pusher::skip_emission(int step) {
  int const emission_start_step = input.kernel.emission_start_step;
  int const emission_interval = input.kernel.emission_interval;
  int const emission_step = step - emission_start_step;
  num_active_emission = int(floor(emission_step/emission_interval));
  bool out_of_wavefront_storage = num_active_emission >= input.kernel.num_wavefronts;
  if (out_of_wavefront_storage) {
     num_active_emission = input.kernel.num_wavefronts - 1;
     if (input.mpi.rank == 0) { 
     std::cout << "num_wavefronts exceeded, emission skipped at step " << step << "\n" << std::flush; 
     }
  }
  return (step < emission_start_step or (emission_step % emission_interval) != 0 or out_of_wavefront_storage);
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr
