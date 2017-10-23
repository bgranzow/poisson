#include "goal_assembly.hpp"
#include "goal_control.hpp"
#include "goal_dbcs.hpp"
#include "goal_disc.hpp"
#include "goal_eval_modes.hpp"
#include "goal_linear_solve.hpp"
#include "goal_poisson.hpp"
#include "goal_primal.hpp"
#include "goal_sol_info.hpp"
#include "goal_soln.hpp"
#include "goal_weight.hpp"

namespace goal {

using Teuchos::rcp;

static void make_soln(Poisson* m, Evaluators& r, Evaluators& j) {
  auto f = m->get_soln();
  auto p = rcp(new Soln<ST>(f, PRIMAL));
  auto pp = rcp(new Soln<FADT>(f, PRIMAL));
  auto w = rcp(new Weight(f));
  r.push_back(p);
  r.push_back(w);
  j.push_back(pp);
  j.push_back(w);
}

Primal::Primal(ParameterList const& p, Poisson* m) {
  params = p;
  poisson = m;
  sol_info = 0;
  make_soln(poisson, residual, jacobian);
  poisson->build_resid<ST>(residual);
  poisson->build_resid<FADT>(jacobian);
}

Primal::~Primal() {
  destroy_data();
}

void Primal::build_data() {
  auto disc = poisson->get_disc();
  sol_info = create_sol_info(disc);
}

void Primal::destroy_data() {
  if (sol_info) {
    destroy_sol_info(sol_info);
    sol_info = 0;
  }
}

void Primal::print_banner(const double t_now) {
  auto ndofs = sol_info->owned->R->getGlobalLength();
  print("*** primal solve: %d dofs", ndofs);
  print("*** at time: %f", t_now);
}

void Primal::compute_jacob(const double t_now, const double t_old) {
  auto t0 = time();
  auto dbc = params.sublist("dirichlet bcs");
  sol_info->resume_fill();
  sol_info->zero_all();
  set_time(jacobian, t_now, t_old);
  assemble(jacobian, sol_info);
  sol_info->gather_all();
  set_jac_dbcs(dbc, sol_info, t_now);
  sol_info->complete_fill();
  auto t1 = time();
  print(" > jacobian computed in %f seconds", t1 - t0);
}

void Primal::solve(const double t_now, const double t_old) {
  print_banner(t_now);
  auto disc = sol_info->get_disc();
  auto R = sol_info->owned->R;
  auto dRdu = sol_info->owned->dRdu;
  auto du = rcp(new VectorT(disc->get_owned_map()));
  auto lp = params.sublist("primal linear algebra");
  compute_jacob(t_now, t_old);
  R->scale(-1.0);
  du->putScalar(0.0);
  goal::solve(lp, dRdu, du, R, disc);
  disc->add_soln(du);
}

Primal* create_primal(ParameterList const& p, Poisson* m) {
  return new Primal(p, m);
}

void destroy_primal(Primal* p) {
  delete p;
}

}
