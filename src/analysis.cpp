#include "analysis.h"
#include <iomanip>

namespace cosyr {

/* -------------------------------------------------------------------------- */
Analysis::Analysis(Input const& input)
  : gamma(input.analysis.gamma),
    coord_shift(input.analysis.shift),
    coord_scale(input.analysis.scale),
    field_scale(input.analysis.field_scale),
    file_exact(input.analysis.file_exact),
    file_error(input.analysis.file_error)
  {
    assert(input.analysis.shift != nullptr);
    assert(input.analysis.scale != nullptr);
    // wavelets should not be restricted to subcycling only
    assert(not input.wavelets.subcycle);
  }

/* -------------------------------------------------------------------------- */
Wonton::vector<double> Analysis::eval(Wonton::Swarm<2> const& points,
                                      std::string const& function) const {

  assert(not function.empty());

  Formula formula;

  if (formula.initialize(function)) {
    // step 1: assess function
    int const num_points = points.num_particles();
    Wonton::vector<double> values(num_points);
    for (int i = 0; i < num_points; ++i) {
      Point<2> const& p = points.get_particle_coordinates(i);
      double const& x = p[0] * coord_scale[0] + coord_shift[0];
      double const& y = p[1] * coord_scale[1] + coord_shift[1];
      values[i] = formula(x, y) * field_scale;
    }

    // step 2: export if requested
    if (not file_exact.empty()) {
      std::ofstream file(file_exact);
      if (file.good()) {
        int const num_axis = static_cast<int>(std::sqrt(num_points));
        for (int i = 0; i < num_axis; ++i) {
          for (int j = 0; j < num_axis; ++j) {
            file << std::setprecision(12);
            file << values[i * num_axis + j] << std::endl;
          }
        }
      } else { throw std::runtime_error("failed to open '"+ file_exact +"'"); }
    }

    return values;
  } else { throw std::runtime_error("invalid analytic function"); }
}

/* -------------------------------------------------------------------------- */
Wonton::vector<double> Analysis::eval(Wonton::Swarm<2> const& points,
                                      std::function<double(double, double)> const& function) const {
  // step 1: assess function
  int const num_points = points.num_particles();
  std::vector<double> values(num_points);
  for (int i = 0; i < num_points; ++i) {
    Point<2> const& p = points.get_particle_coordinates(i);
    double const& x = p[0] * coord_scale[0] + coord_shift[0];
    double const& y = p[1] * coord_scale[1] + coord_shift[1];
    values[i] = function(x, y) * field_scale;
  }

  // step 2: export if requested
  if (not file_exact.empty()) {
    std::ofstream file(file_exact);
    if (file.good()) {
      int const num_axis = static_cast<int>(std::sqrt(num_points));
      for (int i = 0; i < num_axis; ++i) {
        for (int j = 0; j < num_axis; ++j) {
          file << std::setprecision(12);
          file << values[i * num_axis + j] << std::endl;
        }
      }
    } else { throw std::runtime_error("failed to open '"+ file_exact +"'"); }
  }

  return values;
}

/* -------------------------------------------------------------------------- */
double Analysis::error(Wonton::vector<double> const& exact,
                       Wonton::vector<double> const& approx, int norm) const {

  assert(exact.size() == approx.size());

  // step 1: compute error
  int const num_values = exact.size();
  double error[num_values];
  double error_norm = 0.0;

  // store pointwise error
  for (int i = 0; i < num_values; ++i) {
    error[i] = std::abs(exact[i] - approx[i]);
  }

  // compute error in specified norm
  switch (norm) {
    case 0:  for (auto&& value : error) { error_norm = std::max(value, error_norm); } break;
    case 1:  for (auto&& value : error) { error_norm += value; } break;
    case 2:  for (auto&& value : error) { error_norm += value * value; } error_norm = sqrt(error_norm); break;
    default: for (auto&& value : error) { error_norm += std::pow(value, norm); } error_norm = std::pow(error_norm, 1./norm); break;
  }

  // step 2: export if requested
  if (not file_error.empty()) {
    std::ofstream file(file_error);
    if (file.good()) {
      int const num_axis = static_cast<int>(std::sqrt(num_values));
      for (int i = 0; i < num_axis; ++i) {
        for (int j = 0; j < num_axis; ++j) {
          file << std::setprecision(12);
          file << error[i * num_axis + j] << std::endl;
        }
      }
    } else { throw std::runtime_error("failed to open '"+ file_error +"'"); }
  }

  return error_norm;
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr
