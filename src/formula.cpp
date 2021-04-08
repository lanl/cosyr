#include "formula.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
bool Formula::initialize(std::string const& formula) {

  // register variables and symbol table
  symbol_table.add_variable("x", x);
  symbol_table.add_variable("y", y);
  expression.register_symbol_table(symbol_table);

  // parse expression
  if (not parser.compile(formula, expression)) {
    std::fprintf(stderr,
                 "Could not parse the point filter expression '%s'\n"
                 "One culprit could be that one or more variables"
                 "do not match the specified dimension of the problem, "
                 "or the expression itself does not return a boolean.\n"
                 "Please check http://www.partow.net/programming/exprtk "
                 "for more details on syntax description\n",
                 formula.data()
    );
    return false;
  }
  return true;
}

/* -------------------------------------------------------------------------- */
double Formula::operator()(double const& in_x, double const& in_y) {
  x = in_x;
  y = in_y;
  return expression.value();
}

/* -------------------------------------------------------------------------- */
double Formula::operator()(Point<2> const& p) {
  return operator()(p[0], p[1]);
}

/* -------------------------------------------------------------------------- */
} // namespace CSR


