#include "goal_control.hpp"
#include "goal_weight.hpp"

namespace goal {

Weight::Weight(apf::Field* base) {
  GOAL_DEBUG_ASSERT(apf::getValueType(base) == apf::SCALAR);
  shape = apf::getShape(base);
  auto fname = (std::string)apf::getName(base);
  this->name = fname.substr(0, 1) + "w";
}

Weight::~Weight() {
}

ST const& Weight::val(const int node) const {
  return BF[node];
}

ST const& Weight::grad(const int node, const int i) const {
  return GBF[node][i];
}

void Weight::in_elem(apf::MeshElement* me) {
  elem = me;
}

void Weight::at_point(apf::Vector3 const& p, double, double) {
  apf::getBF(shape, elem, p, BF);
  apf::getGradBF(shape, elem, p, GBF);
}

void Weight::out_elem() {
  elem = 0;
}

}
