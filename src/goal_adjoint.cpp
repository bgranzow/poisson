#include <apf.h>
#include <apfMesh2.h>

#include "goal_assembly.hpp"
#include "goal_adjoint.hpp"
#include "goal_control.hpp"
#include "goal_dbcs.hpp"
#include "goal_eval_modes.hpp"
#include "goal_linear_solve.hpp"
#include "goal_poisson.hpp"
#include "goal_nested.hpp"
#include "goal_primal.hpp"
#include "goal_sol_info.hpp"
#include "goal_soln.hpp"
#include "goal_weight.hpp"

namespace goal {

using Teuchos::rcp;

static void make_soln(Disc* d, Evaluators& a) {
  auto m = d->get_apf_mesh();
  auto f = m->findField("u");
  GOAL_DEBUG_ASSERT(f);
  auto p = rcp(new Soln<FADT>(f, ADJOINT));
  auto w = rcp(new Weight(f));
  a.push_back(p);
  a.push_back(w);
}

Adjoint::Adjoint(ParameterList const& p, Primal* pr) {
  params = p;
  primal = pr;
  base_disc = primal->get_poisson()->get_disc();
  nested_disc = create_nested(base_disc, SINGLE);
  auto poisson_params = params.sublist("poisson");
  auto func_params = params.sublist("functional");
  poisson = create_poisson(poisson_params, nested_disc);
  nested_disc->build_data();
  sol_info = create_sol_info(nested_disc);
  make_soln(nested_disc, adjoint);
  poisson->build_resid<FADT>(adjoint);
  poisson->build_functional<FADT>(func_params, adjoint);
}

Adjoint::~Adjoint() {
  destroy_sol_info(sol_info);
  destroy_poisson(poisson);
  destroy_nested(nested_disc);
}

Adjoint* create_adjoint(ParameterList const& p, Primal* pr) {
  return new Adjoint(p, pr);
}

void Adjoint::print_banner(const double t_now) {
  auto ndofs = sol_info->owned->R->getGlobalLength();
  print("**** adjoint solve: %d dofs", ndofs);
  print("**** at time: %f", t_now);
}

void Adjoint::compute_adjoint(const double t_now, const double t_old) {
  auto t0 = time();
  auto dbc = params.sublist("dirichlet bcs");
  sol_info->resume_fill();
  sol_info->zero_all();
  set_time(adjoint, t_now, t_old);
  assemble(adjoint, sol_info);
  sol_info->gather_all();
  set_jac_dbcs(dbc, sol_info, t_now);
  sol_info->complete_fill();
  auto t1 = time();
  print(" > adjoint computed in %f seconds", t1 - t0);
}

void Adjoint::solve(const double t_now, const double t_old) {
  print_banner(t_now);
  auto R = sol_info->owned->R;
  auto dRduT = sol_info->owned->dRdu;
  auto dMdu = sol_info->owned->dMdu;
  auto z = rcp(new VectorT(nested_disc->get_owned_map()));
  auto lp = params.sublist("adjoint linear algebra");
  auto nested_mesh = nested_disc->get_apf_mesh();
  auto zu = apf::createFieldOn(nested_mesh, "zu", apf::SCALAR);
  compute_adjoint(t_now, t_old);
  z->putScalar(0.0);
  goal::solve(lp, dRduT, z, dMdu, nested_disc);
  nested_disc->set_adjoint(z);

  auto err = - (R->dot(*z));
  print("J(u)-J(u^h) ~ %.15e", err);

  apf::writeVtkFiles("debug", nested_disc->get_apf_mesh());
  apf::destroyField(zu);
}

void destroy_adjoint(Adjoint* a) {
  delete a;
}

}
