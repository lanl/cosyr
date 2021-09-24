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
   * @param compute_gradients: whether to compute gradients or not.
   */
  void interpolate(int step, double scaling);

  /**
   * @brief Check whether to remap for current step or not.
   *
   * @param step: current time step
   * @return true if so, false otherwise.
   */
  bool process(int step) const;

private:
  /**
   * @brief Update loaded wavelets for subcycling.
   *
   */
  void update_subcycle_wavelets(bool reset = true);

  /**
   * @brief Update all active wavelets.
   *
   * @param istart: starting index or offset.
   * @param num_active: number of active wavelets.
   */
  void update_active_wavelets(int istart, int num_active);

  /**
   * @brief Collect mesh points and reset state if requested.
   *
   * @param reset: whether to reset the mesh state or not.
   */
  void update_mesh(bool reset = true);

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
  void update_smoothing_lengths(int particle);

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
   * @brief Estimate gradient of mesh fields using a least-squares fit approximation.
   *
   * For a given mesh point and a given field, it solves the normal
   * equation (A^T.A).X = (A^T.F) where A is the matrix of distances between
   * the current point and each of its neighbor, and F is the vector of
   * differences between field values at the current point and that of
   * each of his neighbor.
   */
  void estimate_gradients();

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
   * @brief Unscaled axis lengths for support of weight functions.
   *
   */
  std::vector<double> h_unscaled = {1.0, 1.0};

  /**
   * @brief Smoothing lengths.
   *
   */
  Wonton::vector<Matrix> smoothing_lengths;

  /**
   * @brief Search radii for each mesh point.
   *
   */
  Wonton::vector<Point<DIM>> extents;

  /**
   * @brief Search extents when gathering neighbors for gradient estimation.
   *
   */
  Wonton::vector<Point<DIM>> radii;

  /**
   * @brief List of wavelets in the vicinity of each mesh point.
   *
   */
  Wonton::vector<std::vector<int>> neighbors;

  /**
   * @brief Shape functions type on each mesh point.
   *
   */
  Wonton::vector<Weight::Kernel> kernels;

  /**
   * @brief Support of shape functions on each mesh point.
   *
   */
  Wonton::vector<Weight::Geometry> support;

  /**
   * @brief Remap weights.
   *
   */
  Wonton::vector<std::vector<Wonton::Weights_t>> weights;

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

