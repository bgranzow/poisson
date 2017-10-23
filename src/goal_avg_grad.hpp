#ifndef goal_avg_grad_hpp
#define goal_avg_grad_hpp

#include "goal_qoi.hpp"

namespace goal {

using Teuchos::RCP;
using Teuchos::ParameterList;

class Integrator;
template <typename T> class Soln;

template <typename T>
class AvgGrad : public QoI<T> {
  public:
    AvgGrad(RCP<Integrator> u);
    void at_point(apf::Vector3 const&, double w, double dv);
  private:
    RCP<Soln<T>> u;
    int num_dims;
};

}

#endif
