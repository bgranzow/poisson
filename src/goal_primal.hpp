#ifndef goal_primal_hpp
#define goal_primal_hpp

#include <Teuchos_ParameterList.hpp>

namespace goal {

class Integrator;
class Poisson;
class SolInfo;

using Teuchos::RCP;
using Teuchos::ParameterList;
using Evaluators = std::vector<RCP<Integrator>>;

class Primal {
  public:
    Primal(ParameterList const& p, Poisson* m);
    ~Primal();
    Poisson* get_poisson() { return poisson; }
    SolInfo* get_sol_info() { return sol_info; }
    void build_data();
    void destroy_data();
    void solve(const double t_now, const double t_old);
  private:
    void print_banner(const double t_now);
    void compute_jacob(const double t_now, const double t_old);
    ParameterList params;
    Poisson* poisson;
    SolInfo* sol_info;
    Evaluators residual;
    Evaluators jacobian;
};

Primal* create_primal(ParameterList const& p, Poisson* m);
void destroy_primal(Primal* p);

}

#endif
