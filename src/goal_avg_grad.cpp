#include "goal_avg_grad.hpp"
#include "goal_soln.hpp"

namespace goal {

using Teuchos::rcp_static_cast;

template <typename T>
AvgGrad<T>::AvgGrad(RCP<Integrator> u_) :
    u(rcp_static_cast<Soln<T>>(u_)),
    num_dims(u->get_num_dims()) {
  this->name = "avg grad";
}

template <typename T>
void AvgGrad<T>::at_point(apf::Vector3 const&, double w, double dv) {
  for (int i = 0; i < num_dims; ++i)
    this->elem_value += u->grad(i) * w * dv;
  this->elem_value /= num_dims;
}

template class AvgGrad<ST>;
template class AvgGrad<FADT>;

}
