#pragma once

#include "cosyr.h"
#include "formula.h"
#include "input.h"
#include "wonton/swarm/swarm.h"

namespace cosyr {

/**
 * @brief Helper class for numerical analysis.
 *
 */
class Analysis {
public:
  /**
   * @brief Create an instance.
   *
   * @param input: cached simulation parameters
   */
  explicit Analysis(Input const& input);

  /**
   * @brief Delete the current instance.
   *
   */
  ~Analysis() = default;

  /**
   * @brief Assess analytical field values on a given set of points.
   *
   * @param points: the given set of points.
   * @param function: the analytical expression to evaluate.
   * @return analytical field values.
   */
  Wonton::vector<double> eval(Wonton::Swarm<2> const& points,
                              std::string const& function) const;

  /**
   * @brief Assess analytical field values on a given set of points.
   *
   * @param points: the given set of points.
   * @param function: the given lambda function to evaluate.
   * @return analytical field values.
   */
  Wonton::vector<double> eval(Wonton::Swarm<2> const& points,
                              std::function<double(double, double)> const& function) const;

  /**
   * @brief Assess error of approximated field.
   *
   * @param exact: the exact field values.
   * @param approx: the approximateed field values.
   * @param norm: the norn to consider.
   * @return error field.
   */
  double error(Wonton::vector<double> const& exact,
               Wonton::vector<double> const& approx,
               int norm = 2) const;

public:
  /**
   * @brief Lorentz factor for rescaling.
   *
   */
  double gamma = 1000.;

  /**
   * @brief Coordinates shift terms (x,y).
   *
   */
  double* coord_shift = nullptr;

  /**
   * @brief Coordinates scale factors (x,y).
   *
   */
  double* coord_scale = nullptr;

  /**
   * @brief Field values scale factor.
   *
   */
  double field_scale = 1.0;

  /**
   * @brief Path where to export exact values.
   *
   */
  std::string file_exact;

  /**
   * @brief Path where to export error map.
   *
   */
  std::string file_error;
};

} // namespace cosyr