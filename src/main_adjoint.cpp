#include <Teuchos_YamlParameterListHelpers.hpp>

#include "goal_adjoint.hpp"
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
  p.set<std::string>("adjoint mode", "");
  p.sublist("discretization");
  p.sublist("dirichlet bcs");
  p.sublist("poisson");
  p.sublist("functional");
  p.sublist("primal linear algebra");
  p.sublist("adjoint linear algebra");
  p.sublist("output");
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
    Poisson* poisson;
    Primal* primal;
    Functional* functional;
    Output* output;
};

Solver::Solver(const char* in) {
  print("reading input: %s", in);
  params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, params.ptr());
  params->validateParameters(get_valid_params(), 0);
  auto disc_params = params->sublist("discretization");
  auto poisson_params = params->sublist("poisson");
  auto out_params = params->sublist("output");
  disc = create_disc(disc_params);
  poisson = create_poisson(poisson_params, disc);
  primal = create_primal(*params, poisson);
  functional = create_functional(*params, primal);
  output = create_output(out_params, disc);
}

Solver::~Solver() {
  destroy_output(output);
  destroy_functional(functional);
  destroy_primal(primal);
  destroy_poisson(poisson);
  destroy_disc(disc);
}

void Solver::solve() {
  disc->build_data();
  primal->build_data();
  primal->solve(0.0, 0.0);
  functional->compute(0.0, 0.0);
  functional->print_value();
  primal->destroy_data();
  disc->destroy_data();
  auto adjoint = create_adjoint(*params, primal);
  adjoint->solve(0.0, 0.0);
  destroy_adjoint(adjoint);
  output->write(0.0, 0);
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
