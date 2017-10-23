#include <apfMesh.h>
#include <apfShape.h>

#include "goal_control.hpp"
#include "goal_soln_adjoint.hpp"

namespace goal {

SolnAdjoint::SolnAdjoint(apf::Field* f, apf::Field* c) :
    Weight(c) {
  fine = f;
  coarse = c;
  GOAL_DEBUG_ASSERT(apf::getValueType(fine) == apf::SCALAR);
  GOAL_DEBUG_ASSERT(apf::getValueType(coarse) == apf::SCALAR);
  num_dims = apf::getMesh(coarse)->getDimension();
  num_nodes = 0;
}

ST const& SolnAdjoint::val(const int node) const {
  return values[node];
}

ST const& SolnAdjoint::grad(const int node, const int i) const {
  return gradients[node][i];
}

void SolnAdjoint::in_elem(apf::MeshElement* me) {
  elem = me;
  auto m = apf::getMesh(coarse);
  auto ent = apf::getMeshEntity(elem);
  auto type = m->getType(ent);
  num_nodes = shape->getEntityShape(type)->countNodes();
  coarse_elem = apf::createElement(coarse, elem);
  fine_elem = apf::createElement(fine, elem);
  values.allocate(num_nodes);
  gradients.allocate(num_nodes);
}

void SolnAdjoint::at_point(apf::Vector3 const& p, double, double) {
  apf::Vector3 grad_z;
  apf::Vector3 grad_z_fine;
  apf::getBF(shape, elem, p, BF);
  apf::getGradBF(shape, elem, p, GBF);
  double z = apf::getScalar(coarse_elem, p);
  double z_fine = apf::getScalar(fine_elem, p);
  apf::getGrad(coarse_elem, p, grad_z);
  apf::getGrad(fine_elem, p, grad_z_fine);
  for (int node = 0; node < num_nodes; ++node) {
    values[node] = (z_fine - z) * BF[node];
    for (int i = 0; i < num_dims; ++i)
      gradients[node][i] =
        (grad_z_fine[i] - grad_z[i]) * BF[node] +
        (z_fine - z) * GBF[node][i];
  }
}

void SolnAdjoint::out_elem() {
  elem = 0;
}

}
