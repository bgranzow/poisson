#ifndef goal_avg_soln_hpp
#define goal_avg_soln_hpp

#include "goal_qoi.hpp"

namespace goal {

using Teuchos::RCP;
using Teuchos::ParameterList;

class Integrator;
template <typename T> class Soln;

template <typename T>
class AvgSoln : public QoI<T> {
  public:
    AvgSoln(RCP<Integrator> u);
    void at_point(apf::Vector3 const&, double w, double dv);
  private:
    RCP<Soln<T>> u;
};

}

#endif
