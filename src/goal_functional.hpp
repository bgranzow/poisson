#ifndef goal_functional_hpp
#define goal_functional_hpp

#include <Teuchos_ParameterList.hpp>
#include "goal_scalar_types.hpp"

namespace goal {

class Integrator;
class Primal;
class Poisson;
class SolInfo;
template <typename T> class QoI;

using Teuchos::RCP;
using Teuchos::ParameterList;
using Evaluators = std::vector<RCP<Integrator>>;

class Functional {
  public:
    Functional(ParameterList const& p, Primal* pr);
    ~Functional();
    double get_value();
    void print_value();
    void compute(const double t_now, const double t_old);
  private:
    ParameterList params;
    Primal* primal;
    Poisson* poisson;
    SolInfo* sol_info;
    RCP<QoI<ST>> functional;
    Evaluators evaluators;
};

Functional* create_functional(ParameterList const& p, Primal* pr);
void destroy_functional(Functional* f);

}

#endif
