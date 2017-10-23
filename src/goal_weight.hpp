#ifndef goal_weight_hpp
#define goal_weight_hpp

#include <apf.h>
#include "goal_integrator.hpp"
#include "goal_scalar_types.hpp"

namespace goal {

class Weight : public Integrator {
  public:
    Weight(apf::Field* base);
    virtual ~Weight();
    virtual ST const& val(const int node) const;
    virtual ST const& grad(const int node, const int i) const;
    virtual void in_elem(apf::MeshElement* me);
    virtual void at_point(apf::Vector3 const& p, double, double);
    virtual void out_elem();
  protected:
    apf::FieldShape* shape;
    apf::MeshElement* elem;
    apf::NewArray<ST> BF;
    apf::NewArray<apf::Vector3> GBF;
};

}

#endif
