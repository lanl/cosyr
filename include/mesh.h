#pragma once

#include <iomanip>
#include "cosyr.h"
#include "input.h"

namespace cosyr {

/**
 * @class Moving mesh.
 *  
 */
class Mesh {
public:

#if DIM == 3
  using Coords = Cabana::MemberTypes<double, double, double>;
  using Fields = Cabana::MemberTypes<double, double, double, double>;
#else
  using Coords = Cabana::MemberTypes<double, double>;
  using Fields = Cabana::MemberTypes<double, double, double>;
#endif

  /**
   * @brief Total number of points.
   *
   */
  int num_points = 0;

  /**
   * @brief Number of points in each axis.
   *
   */
  int resolution[DIM] = {0};

  /**
   * @brief Dimensions in unit of normalization length
   *
   */
  double dimension[DIM] = {0.};

  /**
   * @brief Half-length per axis
   *
   */
  double half_width[DIM] = {0.};

  /**
   * @brief Cell size in each axis.
   *
   */
  double h[DIM] = {0.};

  /**
   * @brief Mesh center quantities, in global cartesian coordinates.
   *
   */
  struct {
    /** position of mesh center */
    double position[DIM] = {0.};
    /** velocity of mesh center */
    double velocity[DIM] = {0.};
    /** proper speed of mesh center */
    double speed = 0.;
    /** distance to origin */
    double distance = 0.;
    /** center angles: counterclockwise from +x axis and from x-y plane */
    double angle[DIM - 1] = {0.};
    /** velocity angles */
    double direction[DIM - 1] = {0.};
    /** cached sinus of center angles */
    double sinus_angle[DIM - 1] = {0.};
    /** cached cosine of center angles */
    double cosin_angle[DIM - 1] = {0.};
  } center;

  /**
   * @brief whether the mesh has been updated or not.
   *
   */
  bool is_updated = false;

  /**
   * @brief Number of fields at each point.
   */
  int num_fields = DIM + 1;

  /**
   * @brief Points coordinates: normal|vertical|horizontal
   *
   */
  Cabana::AoSoA<Coords, HostSpace> points;

  /**
   * @brief Fields values at each point
   *
   */
  Cabana::AoSoA<Fields, HostSpace> fields;

  /**
   * @brief Gradients of remapped field at each point.
   *
   */
  Cabana::AoSoA<Fields, HostSpace> gradients[DIM];

  /**
   * @brief Create a new moving mesh.
   *
   * @param input: cached input parameters.
   */
  explicit Mesh(Input const& in_input);

  /**
   * @brief Update mesh position.
   *
   * @param new_position: the coordinates
   * @param new_velocity: the proper velocities
   */
  void move(const double* new_position, const double* new_velocity);

  /**
   * @brief Accumulate fields.
   *
   */
  void sync();

  /**
   * @brief Retrieve data slices corresponding to field index.
   *
   * @tparam AoSoA_t: type of array.
   * @param array: the given array.
   * @param index: the field index.
   * @return the corresponding slice.
   */
  template<class AoSoA_t>
  auto get_slice(AoSoA_t const& array, int index) {
    switch (index) {
      case 0: return Cabana::slice<F1>(array);
      case 1: return Cabana::slice<F2>(array);
      case 2: return Cabana::slice<F3>(array);
  #if DIM == 3
      case 3: return Cabana::slice<F4>(array);
  #endif
      default: throw std::runtime_error("invalid field index");
    }
  }

private:
  /**
   * @brief Reference to simulation parameters.
   *
   */
  Input const& input;
};

} // namespace CSR
