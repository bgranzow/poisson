#ifndef goal_soln_adjoint_hpp
#define goal_soln_adjoint_hpp

#include "goal_weight.hpp"

namespace goal {

class SolnAdjoint : public Weight {
  public:
    SolnAdjoint(apf::Field* fine, apf::Field* coarse);
    ST const& val(const int node) const;
    ST const& grad(const int node, const int i) const;
    void in_elem(apf::MeshElement* me);
    void at_point(apf::Vector3 const& p, double, double);
    void out_elem();
  private:
    int num_dims;
    int num_nodes;
    apf::Field* fine;
    apf::Field* coarse;
    apf::Element* fine_elem;
    apf::Element* coarse_elem;
    apf::NewArray<ST> values;
    apf::NewArray<apf::Vector3> gradients;
};

}

#endif
