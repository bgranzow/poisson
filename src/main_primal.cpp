#include <Teuchos_YamlParameterListHelpers.hpp>

#include "goal_control.hpp"
#include "goal_disc.hpp"
#include "goal_functional.hpp"
#include "goal_poisson.hpp"
#include "goal_output.hpp"
#include "goal_primal.hpp"
#include "goal_regression.hpp"

namespace goal {

using Teuchos::RCP;
using Teuchos::ParameterList;

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("discretization");
  p.sublist("dirichlet bcs");
  p.sublist("poisson");
  p.sublist("functional");
  p.sublist("primal linear algebra");
  p.sublist("output");
  p.sublist("regression");
  return p;
}

class Solver {
  public:
    Solver(const char* in);
    ~Solver();
    void solve();
  private:
    RCP<ParameterList> params;
    Disc* disc;
    Poisson* mech;
    Primal* primal;
    Output* output;
};

Solver::Solver(const char* in) {
  print("reading input: %s", in);
  params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, params.ptr());
  params->validateParameters(get_valid_params(), 0);
  auto disc_params = params->sublist("discretization");
  auto mech_params = params->sublist("poisson");
  auto out_params = params->sublist("output");
  disc = create_disc(disc_params);
  mech = create_poisson(mech_params, disc);
  primal = create_primal(*params, mech);
  output = create_output(out_params, disc);
}

Solver::~Solver() {
  destroy_output(output);
  destroy_primal(primal);
  destroy_poisson(mech);
  destroy_disc(disc);
}

void Solver::solve() {
  disc->build_data();
  primal->build_data();
  auto functional = create_functional(*params, primal);
  primal->solve(0, 0);
  functional->compute(0, 0);
  functional->print_value();
  output->write(0, 0);
  check_J_regression(*params, functional);
  destroy_functional(functional);
}

}

int main(int argc, char** argv) {
  goal::initialize();
  GOAL_DEBUG_ASSERT(argc == 2);
  const char* in = argv[1];
  { goal::Solver solver(in);
    solver.solve(); }
  goal::finalize();
}
