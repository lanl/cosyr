#pragma once

#include "cosyr.h"
#include "beam.h"

namespace cosyr {

class Pusher {
public:
  /**
   * @brief Create an instance of particle pusher.
   *
   * @param in_input: simulation parameters.
   * @param in_beam: electron beam.
   * @param in_mesh: moving mesh.
   * @param in_timer: timer.
   */
  Pusher(Input& in_input, Beam& in_beam, Mesh& in_mesh, Timer& in_timer);

  /**
   * @brief Delete the current instance.
   */
  ~Pusher() = default;

  /**
   * @brief Move particles, wavelets and mesh at given step.
   *
   * @param step: current step.
   * @param t: current time.
   */
  void move(int step, double t);

  /**
   * @brief Update wavefront emission quantities.
   *
   * @param step: current step.
   */
  void update_emission_info();

  /**
   * @brief Whether to skip wavefront emission or not.
   *
   * @param step: current step.
   * @return true if skipped, false otherwise.
   */
  bool skip_emission(int step);

  /**
   * @brief number of active emissions .
   */
  int num_active_emission;

private:

  /**
   * @brief Reference to simulation parameters.
   *
   */
  Input& input;

  /**
   * @brief Reference to electron beam.
   *
   */
  Beam& beam;

  /**
   * @brief Reference to moving mesh.
   *
   */
  Mesh& mesh;

  /**
   * @brief Reference to timer.
   *
   */
  Timer& timer;
};

} // namespace cosyr
