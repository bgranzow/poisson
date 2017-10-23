#include <apfMesh.h>

#include "goal_control.hpp"
#include "goal_disc.hpp"
#include "goal_eval_modes.hpp"
#include "goal_soln.hpp"
#include "goal_sol_info.hpp"

namespace goal {

Soln<ST>::Soln(apf::Field* base, const int mode) {
  disc = 0;
  elem = 0;
  num_dims = 0;
  num_nodes = 0;
  field = base;
  shape = apf::getShape(field);
  num_dims = apf::getMesh(field)->getDimension();
  if (mode == PRIMAL) op = &Soln<ST>::scatter_primal;
  else if (mode == NONE) op = &Soln<ST>::scatter_none;
  else fail("displacement: invalid mode: %d", mode);
  auto fname = (std::string)apf::getName(base);
  this->name = fname.substr(0, 1);
}

Soln<ST>::~Soln() {
}

ST& Soln<ST>::val() {
  return value;
}

ST& Soln<ST>::grad(const int i) {
  return gradient[i];
}

ST& Soln<ST>::nodal(const int n) {
  return node[n];
}

ST& Soln<ST>::resid(const int n) {
  return residual[n];
}

void Soln<ST>::pre_process(SolInfo* s) {
  disc = s->get_disc();
}

void Soln<ST>::gather(apf::MeshElement* me) {
  auto ent = apf::getMeshEntity(me);
  num_nodes = disc->get_num_nodes(ent);
  residual.resize(num_nodes);
  elem = apf::createElement(field, me);
  apf::getScalarNodes(elem, node);
  for (int n = 0; n < num_nodes; ++n)
    residual[n] = 0.0;
}

void Soln<ST>::at_point(apf::Vector3 const& p, double, double) {
  auto me = apf::getMeshElement(elem);
  apf::getBF(shape, me, p, BF);
  apf::getGradBF(shape, me, p, GBF);
  value = nodal(0) * BF[0];
  for (int n = 1; n < num_nodes; ++n)
    value += nodal(n) * BF[n];
  for (int i = 0; i < num_dims; ++i) {
    gradient[i] = nodal(0) * GBF[0][i];
    for (int n = 1; n < num_nodes; ++n)
      gradient[i] += nodal(n) * GBF[n][i];
  }
}

void Soln<ST>::scatter_none(SolInfo*) {
}

void Soln<ST>::scatter_primal(SolInfo* s) {
  auto ent = apf::getMeshEntity(elem);
  auto R = s->ghost->R;
  for (int n = 0; n < num_nodes; ++n) {
    GO row = disc->get_gid(ent, n, 0);
    R->sumIntoGlobalValue(row, resid(n));
  }
}

void Soln<ST>::scatter(SolInfo* s) {
  op(this, s);
  apf::destroyElement(elem);
  elem = 0;
}

void Soln<ST>::post_process(SolInfo*) {
  disc = 0;
}

Soln<FADT>::Soln(apf::Field* base, const int mode) {
  disc = 0;
  elem = 0;
  num_dims = 0;
  num_nodes = 0;
  num_dofs = 0;
  field = base;
  shape = apf::getShape(field);
  num_dims = apf::getMesh(field)->getDimension();
  if (mode == NONE) op = &Soln<FADT>::scatter_none;
  else if (mode == PRIMAL) op = &Soln<FADT>::scatter_primal;
  else if (mode == ADJOINT) op = &Soln<FADT>::scatter_adjoint;
  else fail("displacement: invalid mode: %d", mode);
  auto fname = (std::string)apf::getName(base);
  this->name = fname.substr(0, 1);
}

Soln<FADT>::~Soln() {
}

FADT& Soln<FADT>::val() {
  return value;
}

FADT& Soln<FADT>::grad(const int i) {
  return gradient[i];
}

FADT& Soln<FADT>::nodal(const int n) {
  return node_fadt[n];
}

FADT& Soln<FADT>::resid(const int n) {
  return residual[n];
}

void Soln<FADT>::pre_process(SolInfo* s) {
  disc = s->get_disc();
  gradient.resize(num_dims);
}

void Soln<FADT>::gather(apf::MeshElement* me) {
  auto ent = apf::getMeshEntity(me);
  num_nodes = disc->get_num_nodes(ent);
  num_dofs = disc->get_num_dofs(ent);
  node_fadt.resize(num_nodes);
  residual.resize(num_nodes);
  elem = apf::createElement(field, me);
  apf::getScalarNodes(elem, node_st);
  for (int n = 0; n < num_nodes; ++n) {
    nodal(n).diff(n, num_dofs);
    nodal(n).val() = node_st[n];
    resid(n) = 0.0;
  }
}

void Soln<FADT>::at_point(apf::Vector3 const& p, double, double) {
  auto me = apf::getMeshElement(elem);
  apf::getBF(shape, me, p, BF);
  apf::getGradBF(shape, me, p, GBF);
  value = nodal(0) * BF[0];
  for (int n = 1; n < num_nodes; ++n)
    value += nodal(n) * BF[n];
  for (int i = 0; i < num_dims; ++i) {
    gradient[i] = nodal(0) * GBF[0][i];
    for (int n = 1; n < num_nodes; ++n)
      gradient[i] += nodal(n) * GBF[n][i];
  }
}

void Soln<FADT>::scatter_none(SolInfo*) {
}

void Soln<FADT>::scatter_primal(SolInfo* s) {
  using Teuchos::arrayView;
  auto ent = apf::getMeshEntity(elem);
  auto R = s->ghost->R;
  auto dRdu = s->ghost->dRdu;
  std::vector<GO> cols(num_dofs);
  disc->get_gids(ent, cols);
  auto c = arrayView(&cols[0], num_dofs);
  for (int n = 0; n < num_nodes; ++n) {
    auto v = resid(n);
    auto view = arrayView(&(v.fastAccessDx(0)), num_dofs);
    GO row = disc->get_gid(ent, n, 0);
    R->sumIntoGlobalValue(row, v.val());
    dRdu->sumIntoGlobalValues(row, c, view, num_dofs);
  }
}

void Soln<FADT>::scatter_adjoint(SolInfo* s) {
  using Teuchos::arrayView;
  auto ent = apf::getMeshEntity(elem);
  auto R = s->ghost->R;
  auto dRduT = s->ghost->dRdu;
  std::vector<GO> cols(num_dofs);
  disc->get_gids(ent, cols);
  for (int n = 0; n < num_nodes; ++n) {
    auto v = resid(n);
    auto view = arrayView(&(v.fastAccessDx(0)), num_dofs);
    GO row = disc->get_gid(ent, n, 0);
    R->sumIntoGlobalValue(row, v.val());
    for (int dof = 0; dof < num_dofs; ++dof)
      dRduT->sumIntoGlobalValues(
          cols[dof], arrayView(&row, 1), arrayView(&view[dof], 1));
  }
}

void Soln<FADT>::scatter(SolInfo* s) {
  op(this, s);
  apf::destroyElement(elem);
  elem = 0;
}

void Soln<FADT>::post_process(SolInfo*) {
  disc = 0;
}

}
