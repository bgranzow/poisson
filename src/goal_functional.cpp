#include "goal_assembly.hpp"
#include "goal_control.hpp"
#include "goal_eval_modes.hpp"
#include "goal_functional.hpp"
#include "goal_poisson.hpp"
#include "goal_primal.hpp"
#include "goal_soln.hpp"
#include "goal_qoi.hpp"
#include "goal_weight.hpp"

namespace goal {

using Teuchos::rcp;
using Teuchos::rcp_static_cast;

static void make_soln(Poisson* m, Evaluators& e) {
  auto f = m->get_soln();
  auto u = rcp(new Soln<ST>(f, NONE));
  auto w = rcp(new Weight(f));
  e.push_back(u);
  e.push_back(w);
}

Functional::Functional(ParameterList const& p, Primal* pr) {
  params = p;
  primal = pr;
  poisson = primal->get_poisson();
  sol_info = 0;
  auto fp = params.sublist("functional");
  make_soln(poisson, evaluators);
  poisson->build_functional<ST>(fp, evaluators);
  auto eval = evaluators.back();
  functional = rcp_static_cast<QoI<ST>>(eval);
}

Functional::~Functional() {
}

double Functional::get_value() {
  return functional->get_qoi_value();
}

void Functional::print_value() {
  auto n = functional->get_name();
  auto J = functional->get_qoi_value();
  print(" > functional : %s", n.c_str());
  print(" > J(uH) = %.15e", J);
}

void Functional::compute(const double t_now, const double t_old) {
  auto t0 = time();
  sol_info = primal->get_sol_info();
  set_time(evaluators, t_now, t_old);
  assemble(evaluators, sol_info);
  sol_info = 0;
  auto t1 = time();
  print(" > functional computed in %f seconds", t1 - t0);
};

Functional* create_functional(ParameterList const& p, Primal* pr) {
  return new Functional(p, pr);
}

void destroy_functional(Functional* f) {
  delete f;
}

}
