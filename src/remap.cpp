#include "remap.h"

#include <memory>
#include <lsfits.h>

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
  if (input.remap.scatter) {
    throw std::runtime_error("scatter weights form not supported yet");
  }

  num_fields = std::min(input.wavelets.num_fields, DIM + 1);

  int const num_points = mesh.num_points;
  grid.resize(mesh.num_points);
  extents.resize(num_points);
  neighbors.resize(num_points);
  kernels.resize(num_points, Weight::B4);
  support.resize(num_points, Weight::ELLIPTIC);
  weights.resize(num_points);
  smoothing_lengths.resize(num_points);
  gradients.resize(num_points);
  stencils.resize(num_points);

  // approximate cell sizes in circumferential and radial directions.
  h_unscaled[0] = input.kernel.radius * input.mesh.span_angle / (input.mesh.num_hor - 1);
  h_unscaled[1] = input.mesh.width / (input.mesh.num_ver - 1);

  if (input.mpi.rank == 0) {
    std::cout << "num_fields: " << num_fields << std::endl;
    std::cout << "remap: hr: "<< h_unscaled[1] << ", hc: " << h_unscaled[0] << std::endl;
  }

  // set smoothing lengths
  h[0] = input.remap.scaling[0] * h_unscaled[0];
  h[1] = input.remap.scaling[1] * h_unscaled[1];

  Kokkos::parallel_for(HostRange(0, num_points),
                       [&](int i) { smoothing_lengths[i] = Matrix(1, h); });

  // set search radii
  Kokkos::parallel_for(HostRange(0, num_points),
                       [&](int i) { extents[i] = {h[0], h[1] }; });

  gamma = input.kernel.motion_params[0];
  dtdx = input.kernel.dt / h_unscaled[0];

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
void Remap::collect_grid(bool reset) {

  // Create the particle swarm corresponding to the mesh vertices
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

  if (reset) {
    // initialize the state whose values will be populated after remap.
    target.init(grid);
    for (int i = 0; i < num_fields; i++) {
      target.add_field<double>(fields[i], 0.0);
    }
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
void Remap::update_smoothing_lengths(int particle) {

  static_assert(DIM == 2, "dimension not yet supported");

  if (input.remap.adaptive) {
    double const one_third = 1./3.;
    int const num_points = mesh.num_points;

    // deduce the offset to the mesh point coordinate from the current particle
    // position when computing the adaptive smoothing lengths.
    auto const offset = deduce_local_coords(particle);

    // offset on longitudinal coordinates to avoid the spiky region
    double const alpha_min = 12e-6;
    // scaling factor for smoothing length to cover the right end of the domain
    double const h_scaling = 1.5 * 225.;

    Kokkos::parallel_for(HostRange(0, num_points), [&](int i) {
      auto const p = grid.get_particle_coordinates(i);
      auto h_adap = h;
      double const alpha = p[0] - offset[0];
      if (alpha > alpha_min) {
        double const psi = std::pow(24. * alpha, one_third);
        h_adap[0] = h_scaling * h[0] *
          (std::pow(psi, 3.)/6. + psi/gamma/gamma - alpha)
          / (alpha + psi);
      }
      smoothing_lengths[i] = Matrix(1, h_adap);
      extents[i] = { h_adap[0], h_adap[1] };
    });
  }
}

/* -------------------------------------------------------------------------- */
void Remap::run(int particle, bool accumulate, bool rescale, double scaling) {

  using Filter = Portage::SearchPointsBins<DIM, Wonton::Swarm<DIM>, Wonton::Swarm<DIM>>;
  using Accumulator = Portage::Accumulate<DIM, Wonton::Swarm<DIM>, Wonton::Swarm<DIM>>;
  using Estimator = Portage::Estimate<DIM, Wonton::SwarmState<DIM>>;

#if REPORT_TIME
  float elapsed[4] = {0, 0, 0, 0};
  auto tic = timer::now();
#endif

  if (input.remap.adaptive) {
    update_smoothing_lengths(particle);
    #if REPORT_TIME
      elapsed[0] = timer::elapsed(tic, true);
    #endif
  }

  Filter search(wave, grid, extents, extents, WeightCenter::Gather, 1);
  Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    neighbors.begin(), search);
#if REPORT_TIME
  elapsed[1] = timer::elapsed(tic, true);
#endif

  // compute remap weights
  Accumulator accumulator(wave, grid, Portage::LocalRegression, WeightCenter::Gather,
                          kernels, support, smoothing_lengths, basis::Unitary);
  Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    neighbors.begin(), weights.begin(), accumulator);
#if REPORT_TIME
  elapsed[2] = timer::elapsed(tic, true);
#endif

  // estimate field on mesh points
  Estimator estimator(source);

  for (auto const& current : fields) {
    estimator.set_variable(current);
    auto& values = target.get_field(current);
    Wonton::pointer<double> field(values.data());
    Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      weights.begin(), field, estimator);
  }
#if REPORT_TIME
  elapsed[3] = timer::elapsed(tic);

  std::cout << "Remap time: " << elapsed[0] + elapsed[1] + elapsed[2] + elapsed[3] << " (s)" << std::endl;
  std::cout << " - smoothing lengths: " << elapsed[0] << " (s)" << std::endl;
  std::cout << " - filter neighbors: " << elapsed[1] << " (s)" << std::endl;
  std::cout << " - compute weights: " << elapsed[2] << " (s)" << std::endl;
  std::cout << " - estimate fields: " << elapsed[3] << " (s)" << std::endl;
#endif

  // copy back values to mesh
  auto const range = HostRange(0, mesh.num_points);

  for (int i = 0; i < num_fields; i++) {
    auto slice = mesh.get_field_slice(mesh.fields, i);
    auto const& values = target.get_field(fields[i]);

    if (accumulate) {
      Kokkos::parallel_for(range, [&](int j) { slice(j) += values[j]; });
    } else {
      Kokkos::parallel_for(range, [&](int j) { slice(j) = values[j]; });
    }
    if (rescale) {
      Kokkos::parallel_for(range, [&](int j) { slice(j) *= scaling; });
    }
  }
}

/* -------------------------------------------------------------------------- */
void Remap::estimate_gradients() {

  // step 0: update mesh points coordinates
  collect_grid(false);

  // step 1: retrieve the neighbors of each mesh point
  using Filter = Portage::SearchPointsBins<DIM, Wonton::Swarm<DIM>, Wonton::Swarm<DIM>>;

  Filter search(grid, grid, extents, extents, WeightCenter::Gather, 3);
  Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                    neighbors.begin(), search);

  // enrich each list of neighbors by prepending it with the current point index
  Wonton::for_each(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                   grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                   [&](int i) {
                     std::vector<int> indices = neighbors[i];
                     indices.emplace(indices.begin(), i);
                     neighbors[i] = indices;
                   });


  // retrieve the coordinates of each point of the stencil
  Wonton::vector<std::vector<Point<DIM>>> coordinates(mesh.num_points);
  Wonton::transform(neighbors.begin(), neighbors.end(), coordinates.begin(),
                    [&](auto const& indices) {
                      std::vector<Wonton::Point<DIM>> points;
                      for (int j : indices) {
                        points.emplace_back(grid.get_particle_coordinates(j));
                      }
                      return points;
                    });


  // step 2: build and cache stencil matrices
  Wonton::transform(coordinates.begin(), coordinates.end(), stencils.begin(),
                    [&](auto const& points) {
                      return Wonton::build_gradient_stencil_matrices<DIM>(points, true);
                    });

  Wonton::vector<std::vector<double>> values(mesh.num_points);

  for (int f = 0; f < num_fields; ++f) {

    auto const field = mesh.get_field_slice(mesh.fields, f);

    // retrieve field values
    Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      values.begin(),
                      [&](int i) {
                        std::vector<int> const& indices = neighbors[i];
                        std::vector<double> data;
                        data.reserve(indices.size());
                        for (int j : indices) { data.emplace_back(field(j)); }
                        return data;
                      });

    // build right-hand sides and solve equations
    Wonton::transform(grid.begin(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      grid.end(Wonton::PARTICLE, Wonton::PARALLEL_OWNED),
                      gradients.begin(),
                      [&](int i) {
                        std::vector<Wonton::Matrix> const& stencil = stencils[i];
                        return Wonton::ls_gradient<DIM>(stencil[0], stencil[1], values[i]);
                      });

    // store back to mesh
    auto dx = Cabana::slice<X>(mesh.gradients[f]);
    auto dy = Cabana::slice<Y>(mesh.gradients[f]);
    Kokkos::parallel_for(HostRange(0, mesh.num_points),
                         [&](int j) {
                           Wonton::Vector<DIM> const nabla = gradients[j];
                           dx(j) = nabla[0];
                           dy(j) = nabla[1];
                         });
  }
}

/* -------------------------------------------------------------------------- */
void Remap::interpolate(int step, double scaling, bool compute_gradients) {

  bool use_loaded_only = input.wavelets.found and not input.wavelets.subcycle;

  if (process(step)) {

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

    timer.start("mesh_sync");
    mesh.sync();
    timer.stop("mesh_sync");

    // compute the gradients once the field is updated
    if (compute_gradients) { estimate_gradients(); }
  }
}


/* -------------------------------------------------------------------------- */
bool Remap::process(int step) const {

  return ((step >= input.remap.start_step)
         and ((step - input.remap.start_step) % input.remap.interval == 0
            or step == input.kernel.num_step - 1));
}


/* -------------------------------------------------------------------------- */
void Remap::print_info(int count_active, int ratio_loaded) const {

  if (input.mpi.rank == 0) {
    if (ratio_loaded < 0 or ratio_loaded > 100) {
      throw std::runtime_error("invalid ratio of loaded wavelets");
    } else {
      std::cout << "interpolate from " << input.kernel.num_particles << " particles, ";
      if (ratio_loaded < 100) {
        std::cout << count_active << " wavelets/particle";
        if (ratio_loaded > 0) {
          std::cout << "("<< ratio_loaded << "% prescribed)";
        }
      } else {
        std::cout << count_active << " loaded wavelets";
      }
      std::cout << " ... " << std::flush;
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


