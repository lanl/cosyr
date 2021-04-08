#pragma once

#include "cosyr.h"
#include "exprtk.hpp"
#include "portage/support/portage.h"
// avoid long namespaces
using Wonton::Point;

namespace cosyr {

/**
 * @brief Assess analytical formula.
 *
 */
class Formula {
public:

  /**
   * @brief Initialize the parser.
   *
   * @param formula: the analytical expression to parse.
   * @return true if everything correctly set, false otherwise.
   */
  bool initialize(std::string const& formula);

  /**
   * @brief Evalute the function on the given point.
   *
   * @param in_x: abscissa of the point.
   * @param in_y: ordinate of the point.
   * @return the value of the function at (in_x, in_y).
   */
  double operator()(double const& in_x, double const& in_y);

  /**
   * @brief Evaluate the function on the given point.
   *
   * @param p: the point.
   * @return the value of the function at p.
   */
  double operator()(Point<2> const& p);

private:
  double x = 0, y = 0;                       /** current point coordinates */
  exprtk::symbol_table<double> symbol_table; /** table of variables */
  exprtk::expression<double> expression;     /** the expression to parse */
  exprtk::parser<double> parser;             /** the actual parser */
};

} // namespace CSR
