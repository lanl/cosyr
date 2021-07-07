#include "remap.h"

#include <memory>

namespace cosyr {

/* -------------------------------------------------------------------------- */
Remap::Remap(Input& in_input,
             Beam& in_beam,
             Wavelets& in_wavelets,
             Mesh& in_mesh,
             Analysis& in_analysis,
             Timer& in_timer)
  : input(in_input),
    beam(in_beam),
    wavelets(in_wavelets),
    mesh(in_mesh),
    analysis(in_analysis),
    timer(in_timer)
{
  num_fields = std::min(input.wavelets.num_fields, DIM + 1);

  // approximate cell sizes in circumferential and radial directions.
  double hc = input.kernel.radius * input.mesh.span_angle / (input.mesh.num_hor - 1);
  double hr = input.mesh.width / (input.mesh.num_ver - 1);

  if (input.mpi.rank == 0) {
    std::cout << "num_fields: " << num_fields << std::endl;
    std::cout << "remap: hr: "<< hr << ", hc: " << hc << std::endl;
  }

  h[0] = input.remap.scaling[0] * hc;
  h[1] = input.remap.scaling[1] * hr;
  gamma = input.kernel.motion_params[0];
  dtdx = input.kernel.dt / hc;

  for (int i = 1; i <= num_fields; ++i) {
    fields.emplace_back("fld" + std::to_string(i));
  }
}

/* -------------------------------------------------------------------------- */
void Remap::collect_subcycle_wavelets() {

  wave.resize(input.wavelets.count);
  Wonton::vector<double> field(input.wavelets.count);

  for (int j = 0; j < input.wavelets.count; j++) {
    #if DIM == 3
      Wonton::Point<DIM> p(input.wavelets.x[j],
                           input.wavelets.y[j],
                           input.wavelets.z[j]);
    #else
      Wonton::Point<DIM> p(input.wavelets.x[j], input.wavelets.y[j]);
    #endif
    wave.assign(j, p);
  }

  // initialize a state manager for the field swarm and populate it
  source.init(wave);

  for (int i = 0; i < num_fields; ++i) {
    int const k = i * input.wavelets.count;
    for (int j = 0; j < input.wavelets.count; j++) {
      field[j] = input.wavelets.field[j + k];
    }
    source.add_field(fields[i], field);
  }
}

/* -------------------------------------------------------------------------- */
void Remap::collect_active_wavelets(int index_particle, int num_active) {

  bool const subcycle = input.wavelets.found and input.wavelets.subcycle;
  int const num_loaded = (subcycle ? input.wavelets.count : 0);

  wave.resize(num_active);
  Wonton::vector<double> field(num_active);

  // step 2: copy loaded wavelets
  if (subcycle) {
    for (int j = 0; j < num_loaded; j++) {
      int const k = j + (index_particle * num_loaded);
      #if DIM == 3
        Wonton::Point<DIM> p(wavelets.loaded.host.coords(k, WT_POS_X),
                             wavelets.loaded.host.coords(k, WT_POS_Y),
                             wavelets.loaded.host.coords(k, WT_POS_Z));
      #else
        Wonton::Point<DIM> p(wavelets.loaded.host.coords(k, WT_POS_X),
                             wavelets.loaded.host.coords(k, WT_POS_Y));
      #endif
      wave.assign(j, p);
    }
  }

  // step 1: copy active emitted wavelets
  int const num_emitted = num_active - num_loaded;
  int const total_emitted = input.kernel.num_wavefronts * input.kernel.num_dirs;
  int const index_start = index_particle * total_emitted;

  for (int j = 0; j < num_emitted; j++) {
    int const k = j + index_start;
    #if DIM == 3
      Wonton::Point<DIM> p(wavelets.emitted.host.coords(k, WT_POS_X),
                           wavelets.emitted.host.coords(k, WT_POS_Y),
                           wavelets.emitted.host.coords(k, WT_POS_Z));
    #else
      Wonton::Point<DIM> p(wavelets.emitted.host.coords(k, WT_POS_X),
                           wavelets.emitted.host.coords(k, WT_POS_Y));
    #endif
    wave.assign(j + num_loaded, p);
  }

  source.init(wave);

  for (int i = 0; i < num_fields; i++) {
    // step 4: copy all loaded fields
    if (subcycle) {
      for (int j = 0; j < num_loaded; ++j) {
        int const k = j + (index_particle * num_loaded);
        field[j] = wavelets.loaded.host.fields[i](k);
      }
    }

    // step 3: copy all active emitted wavelet fields
    for (int j = 0; j < num_emitted; ++j) {
      int const k = j + index_start;
      field[j + num_loaded] = wavelets.emitted.host.fields[i](k);
    }

    source.add_field(fields[i], field);
  }
}

/* -------------------------------------------------------------------------- */
void Remap::collect_grid() {

  // Create the particle swarm corresponding to the mesh vertices
  grid.resize(mesh.num_points);

  auto x = Cabana::slice<X>(mesh.points);
  auto y = Cabana::slice<Y>(mesh.points);
  #if DIM==3
    auto z = Cabana::slice<Z>(mesh.points);
  #endif

  int n = 0;
  for (int i = 0; i < mesh.resolution[0]; i++) {
    for (int j = 0; j < mesh.resolution[1]; j++) {
      int const k = i * mesh.resolution[1] + j;
      #if DIM==3
        Wonton::Point<DIM> p(x(k), y(k), z(k));
      #else
        Wonton::Point<DIM> p(x(k), y(k));
      #endif
      grid.assign(n++, p);
    } 
  }

  // initialize the swarm and state whose values will be populated after remap.
  target.init(grid);
  for (int i = 0; i < num_fields; i++) {
    target.add_field<double>(fields[i], 0.0);
  }
}

/* -------------------------------------------------------------------------- */
Wonton::Point<DIM> Remap::deduce_local_coords(int particle) const {

  static_assert(DIM == 2, "dimension not yet supported");

  // compute local coordinates of current particle
  auto position = Cabana::slice<Beam::Position>(beam.particles);
  double const x = position(particle, PART_POS_X);
  double const y = position(particle, PART_POS_Y);
  double const cos_angle = mesh.center.cosin_angle[0];
  double const sin_angle = mesh.center.sinus_angle[0];
  double const x_local = x * cos_angle - y * sin_angle;
  double const y_local = x * sin_angle + y * cos_angle - input.kernel.radius;
  return { x_local, y_local };
}

/* -------------------------------------------------------------------------- */
Wonton::vector<Remap::Matrix> Remap::compute_smoothing_length(int particle) const {

  static_assert(DIM == 2, "dimension not yet supported");

  auto& swarm = input.remap.scatter ? wave : grid;
  int const num_points = swarm.num_particles();
  double const one_third = 1./3.;
  Wonton::vector<Matrix> result(num_points);

  // deduce the offset to the mesh point coordinate from the current particle
  // position when computing the adaptive smoothing lengths.
  auto const offset = (input.remap.adaptive ? deduce_local_coords(particle)
                                            : Wonton::Point<DIM>(0.,0.));

  // offset on longitudinal coordinates to avoid the spiky region
  double const alpha_min = 12e-6;
  // scaling factor for smoothing length to cover the right end of the domain
  double const h_scaling = 1.5 * 225.;

  Kokkos::parallel_for(HostRange(0, num_points), [&](int i) {
    auto const p = swarm.get_particle_coordinates(i);
    auto h_adap = h;
    double const alpha = p[0] - offset[0];
    if (input.remap.adaptive and alpha > alpha_min) {
      double const psi = std::pow(24. * alpha, one_third);
      h_adap[0] = h_scaling * h[0] *
                  (std::pow(psi, 3.)/6. + psi/gamma/gamma - alpha)
                  / (alpha + psi);
    }
    result[i] = Matrix(1, h_adap);
  });

  return result;
}

/* -------------------------------------------------------------------------- */
void Remap::run(int particle, bool accumulate, bool rescale, double scaling) {

  // regression parameters
  auto const weight_center = input.remap.scatter ? WeightCenter::Scatter
                                                 : WeightCenter::Gather;

  auto smoothing_lengths = compute_smoothing_length(particle);

  // perform the remap
  driver = std::make_unique<Remapper>(wave, source, grid, target,
                                      smoothing_lengths, Weight::B4,
                                      Weight::ELLIPTIC, weight_center);
  driver->set_remap_var_names(fields, fields,LocalRegression, basis::Unitary);
  driver->run(nullptr, input.remap.verbose);

  // copy back values to mesh
#if DIM == 3
  auto all_fields = { Cabana::slice<F1>(mesh.fields),
                      Cabana::slice<F2>(mesh.fields),
                      Cabana::slice<F3>(mesh.fields),
                      Cabana::slice<F4>(mesh.fields) };
#else
  auto all_fields = { Cabana::slice<F1>(mesh.fields),
                      Cabana::slice<F2>(mesh.fields),
                      Cabana::slice<F3>(mesh.fields) };
#endif

  for (int i = 0; i < num_fields; i++) {
    auto field = all_fields.begin()[i];
    auto const& remapped_values = target.get_field(fields[i]);
    int const size = remapped_values.size();

    if (accumulate) {
      for (int j = 0; j < size; ++j) {
        field(j) += remapped_values[j];
      }
    } else {
      for (int j = 0; j < size; ++j) {
        field(j) = remapped_values[j];
      }
    }
    if (rescale) {
      for (int j = 0; j < size; ++j) {
        field(j) *= scaling;
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
void Remap::interpolate(int step, double scaling) {

  bool do_remap = ((step >= input.remap.start_step)
                  and ((step - input.remap.start_step) % input.remap.interval == 0
                    or step == input.kernel.num_step - 1));

  bool use_loaded_only = input.wavelets.found and not input.wavelets.subcycle;

  if (do_remap) {

    auto num_active = wavelets.transfer_to_host();
    bool accumulate = false;
    bool rescale = false;

    timer.start("interpolation");
    if (use_loaded_only) {
      // interpolate the prescribed field at prescribed wavelets
      print_info(input.wavelets.count);
      collect_subcycle_wavelets();
      collect_grid();
      run(0, accumulate, rescale, 1.0);
      print_progress();

      // assess numerical error
      if (input.analysis.set) {
        std::cout << "---------- error analysis --------" << std::endl;
        auto exact  = analysis.eval(grid, input.analysis.field_expr);
        auto approx = target.get_field("fld1");
        auto error  = analysis.error(exact, approx, 2);
        std::cout << "error L2: " << error << ", field scaling: "<< analysis.field_scale << std::endl;
      }
    } else {
      // interpolate field for all particles
      if (step > 0) {

        // skip duplicate reference particles on other ranks
        int const first_particle = (input.mpi.rank == 0 ? 0 : 1);
        int const last_particle  = input.kernel.num_particles - 1;
        int const count_active = num_active[0];
        int const ratio_loaded = static_cast<int>(100 * input.wavelets.count / wavelets.active[0]);

        for (int j = first_particle; j <= last_particle; j++) {
          print_info(count_active, ratio_loaded);
          collect_active_wavelets(j, num_active[j]);
          collect_grid();
          rescale = (j == last_particle);
          run(j, accumulate, rescale, scaling);
          accumulate = true;
          print_progress(j, last_particle);
        }
      }
    }

    MPI_Barrier(input.mpi.comm);
    timer.stop("interpolation");
  }
}

/* -------------------------------------------------------------------------- */
void Remap::print_info(int count_active, int ratio_loaded) const {

  if (input.mpi.rank == 0) {
    if (ratio_loaded < 0 or ratio_loaded > 100) {
      throw std::runtime_error("invalid ratio of loaded wavelets");
    } else if (ratio_loaded < 100) {
      std::cout << "interpolate using "<< count_active << " active wavelets "
                << "("<< ratio_loaded << "% prescribed) ... ";
    } else {
      std::cout << "interpolate using " << count_active
                << " loaded wavelets ... " << std::flush;
    }
  }
}


/* -------------------------------------------------------------------------- */
void Remap::print_progress(int current_particle, int last_particle) const {

  if (input.mpi.rank == 0) {
    if (current_particle < last_particle) {
      int const progress = static_cast<int>(100 * current_particle/ last_particle);
      std::cout << progress << "%\r" << std::flush;
    } else {
      std::cout << "done.\r" << std::endl;
    }
  }
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr


