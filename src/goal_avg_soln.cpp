#include "goal_avg_soln.hpp"
#include "goal_soln.hpp"

namespace goal {

using Teuchos::rcp_static_cast;

template <typename T>
AvgSoln<T>::AvgSoln(RCP<Integrator> u_) :
    u(rcp_static_cast<Soln<T>>(u_)) {
  this->name = "avg soln";
}

template <typename T>
void AvgSoln<T>::at_point(apf::Vector3 const&, double w, double dv) {
  this->elem_value += u->val() * w * dv;
}

template class AvgSoln<ST>;
template class AvgSoln<FADT>;

}
