#ifndef goal_adjoint_hpp
#define goal_adjoint_hpp

#include <Teuchos_ParameterList.hpp>

namespace goal {

class Disc;
class Integrator;
class Poisson;
class Nested;
class Primal;
class SolInfo;

using Teuchos::RCP;
using Teuchos::ParameterList;
using Evaluators = std::vector<RCP<Integrator>>;

class Adjoint {
  public:
    Adjoint(ParameterList const& p, Primal* pr);
    ~Adjoint();
    void solve(const double t_now, const double t_old);
  private:
    void print_banner(const double t_now);
    void compute_adjoint(const double t_now, const double t_old);
    ParameterList params;
    Primal* primal;
    Disc* base_disc;
    Nested* nested_disc;
    Poisson* poisson;
    SolInfo* sol_info;
    Evaluators adjoint;
};

Adjoint* create_adjoint(ParameterList const& p, Primal* pr);
void destroy_adjoint(Adjoint* a);

}

#endif
