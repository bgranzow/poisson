#include <apf.h>
#include <apfMesh2.h>
#include <apfShape.h>

#include "goal_assembly.hpp"
#include "goal_avg_soln.hpp"
#include "goal_avg_grad.hpp"
#include "goal_control.hpp"
#include "goal_disc.hpp"
#include "goal_point_wise.hpp"
#include "goal_poisson.hpp"
#include "goal_residual.hpp"
#include "goal_scalar_types.hpp"
#include "goal_soln.hpp"

namespace goal {

using Teuchos::rcp;

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("f", "");
  return p;
}

Poisson::Poisson(ParameterList const& p, Disc* d) {
  p.validateParameters(get_valid_params(), 0);
  params = p;
  disc = d;
  soln = 0;
  make_soln();
}

Poisson::~Poisson() {
  apf::destroyField(soln);
}

void Poisson::make_soln() {
  auto m = disc->get_apf_mesh();
  if (disc->is_parent()) {
    soln = apf::createFieldOn(m, "u", apf::SCALAR);
    apf::zeroField(soln);
  }
  else {
    soln = m->findField("u");
    GOAL_DEBUG_ASSERT(soln);
  }
}

template <typename T>
void Poisson::build_resid(Evaluators& E) {
  auto f = params.get<std::string>("f");
  auto u = find_evaluator("u", E);
  auto w = find_evaluator("uw", E);
  auto R = rcp(new Residual<T>(u, w, f));
  E.push_back(R);
}

template <typename T>
void Poisson::build_functional(ParameterList const& params, Evaluators& E) {
  auto type = params.get<std::string>("type");
  auto u = find_evaluator("u", E);
  RCP<QoI<T>> J;
  if (type == "avg soln")
    J = rcp(new AvgSoln<T>(u));
  else if (type == "avg grad")
    J = rcp(new AvgGrad<T>(u));
  else if (type == "point wise")
    J = rcp(new PointWise<T>(params));
  else
    fail("unknown functional type: %s", type.c_str());
  E.push_back(J);
}

Poisson* create_poisson(ParameterList const& p, Disc* d) {
  return new Poisson(p, d);
}

void destroy_poisson(Poisson* m) {
  delete m;
}

template void Poisson::build_resid<ST>(Evaluators&);
template void Poisson::build_resid<FADT>(Evaluators&);
template void Poisson::build_functional<ST>(ParameterList const&, Evaluators&);
template void Poisson::build_functional<FADT>(ParameterList const&, Evaluators&);

}
