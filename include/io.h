#pragma once

#include "cosyr.h"
#include "beam.h"

namespace cosyr {

class IO {
public:
  /**
   * @brief Create an instance of I/O manager.
   *
   * @param in_input: cached simulation parameters
   * @param in_beam: electron beam.
   * @param in_mesh: moving mesh.
   * @param in_timer: timer
   */
  IO(Input& in_input,
     Beam& in_beam,
     Wavelets& in_wavelets,
     Mesh& in_mesh,
     Timer& in_timer);

  /**
   * @brief Delete instance.
   *
   */
  ~IO() = default;

  /**
   * @brief Dump simulation data to disk.
   *
   * @param i: current step.
   */
  void dump(int i);

private:
  /**
   * @brief Dump wavelets data into disk.
   *
   * @param step: current step.
   */
  void dump_wavelets(int step) const;

  /**
   * @brief Dump particles data into disk.
   *
   * @param step: current step.
   */
  void dump_particles(int step) const;

  /**
   * @brief Dump mesh data into disk.
   *
   * @param step: current step.
   */
  void dump_mesh(int step) const;

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
   * @brief Reference to wavelets.
   *
   */
  Wavelets& wavelets;

  /**
   * @brief Reference to moving mesh.

   */
  Mesh& mesh;

  /**
   * @brief Reference to timer.
   *
   */
  Timer& timer;
};

} // namespacee cosyr
