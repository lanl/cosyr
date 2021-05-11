#pragma once

#include "cosyr.h"
#include "beam.h"
#include "mesh.h"
#include "analysis.h"

// skip long qualified names
using namespace Portage::Meshfree;

namespace cosyr {

class Remap {
  // alias for matrices.
  using Matrix = std::vector<std::vector<double>>;

public:
  /**
   * @brief Create an instance of remapper.
   *
   * @param in_input: simulation parameters.
   * @param in_wavelets: wavelets.
   * @param in_mesh: moving mesh.
   * @param in_analysis: analyser.
   * @param in_timer: timer.
   */
  Remap(Input& in_input,
        Beam& in_beam,
        Wavelets& in_wavelets,
        Mesh& in_mesh,
        Analysis& in_analysis,
        Timer& in_timer);

  /**
   * @brief Interpolate radiation fields from wavelets to mesh.
   *
   * @param step: current step.
   * @param scaling: field scaling factor.
   */
  void interpolate(int step, double scaling);

  /**
   * @brief Remap interval check.
   *
   */
  bool do_remap(int step);

private:
  /**
   * @brief Collect loaded wavelets for subcycling.
   *
   */
  void collect_subcycle_wavelets();

  /**
   * @brief Collect all active wavelets.
   *
   * @param istart: starting index or offset.
   * @param num_active: number of active wavelets.
   */
  void collect_active_wavelets(int istart, int num_active);

  /**
   * @brief Collect mesh points.
   *
   */
  void collect_grid();

  /**
   * @brief Compute smoothing lengths for each point.
   *
   * It defines the support of a weight function assigned on each point.
   * It is defined on mesh points for gather weight form and on wavelets
   * for scatter weight form. It determines how many near field points
   * will influence the value on a mesh point.
   *
   * @param particle: index of particle emitting the set of wavelets.
   */
  Wonton::vector<Matrix> compute_smoothing_length(int particle) const;

  /**
   * @brief Deduce the local coordinates (x',y') of the current particle.
   *
   * It is used as an offset to the coordinates of each mesh point when
   * computing adaptive smoothing lengths.
   *
   * @param particle: index of particle emitting the set of wavelets.
   * @return
   */
  Wonton::Point<DIM> deduce_local_coords(int particle) const;

  /**
   * @brief Remap the wavelet field values to mesh.
   *
   * @param particle: index of particle emitting the set of wavelets.
   * @param accumulate: whether to accumulate field values or not.
   * @param rescale: whether to rescale field values or not.
   * @param scaling: field scaling factor.
   */
  void run(int particle, bool accumulate, bool rescale, double scaling);

  /**
   * @brief Print remap info.
   *
   * @param count_active: number of active wavelets.
   * @param ratio_loaded: ratio of loaded ones.
   */
  void print_info(int count_active, int ratio_loaded = 100) const;

  /**
   * @brief Print remap progress.
   *
   * @param current: index of current particle.
   * @param last: index of last particle
   */
  void print_progress(int current_particle = 0, int last_particle = 0) const;

  /**
   * @brief Weight function support axis lengths.
   *
   */
  std::vector<double> h = {1.0, 1.0};

  /**
   * @brief Number of fields to remap.
   *
   */
  int num_fields = DIM + 1;

  /**
   * @brief Lorentz factor for rescaling.
   *
   */
  double gamma = 10.0;

  /**
   * @brief Ratio of dt on dx.
   *
   */
  double dtdx = 1.0;

  /**
   * @brief Names of fields to remap.
   *
   */
  std::vector<std::string> fields;

  /**
   * @brief Wavelets.
   *
   */
  Wonton::Swarm<DIM> wave;

  /**
   * @brief Mesh points.
   *
   */
  Wonton::Swarm<DIM> grid;

  /**
   * @brief Near fields from wavelets.
   *
   */
  Wonton::SwarmState<DIM> source;

  /**
   * @brief Remapped fields.
   *
   */
  Wonton::SwarmState<DIM> target;

  /**
   * @brief Remap driver.
   *
   */
  std::unique_ptr<Remapper> driver;

  /**
   * @brief Reference to simulation parameters.
   *
   */
  Input& input;

  /**
   * @brief Reference to beam.
   *
   */
  Beam& beam;

  /**
   * @brief Reference to wavelets.
   *
   */
  Wavelets& wavelets;

  /**
   * @brief Reference to moving mesh.
   *
   */
  Mesh& mesh;

  /**
   * @brief Reference to analysis object.
   *
   */
  Analysis& analysis;

  /**
   * @brief Reference to timer.
   *
   */
  Timer& timer;
};

} // namespace cosyr

