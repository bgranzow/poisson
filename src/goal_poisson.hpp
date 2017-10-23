#ifndef goal_poisson_hpp
#define goal_poisson_hpp

#include <Teuchos_ParameterList.hpp>

namespace apf {
class Field;
}

namespace goal {

class Disc;
class Integrator;
class SolInfo;
class States;

using Teuchos::RCP;
using Teuchos::ParameterList;
using Evaluators = std::vector<RCP<Integrator>>;

class Poisson {
  public:
    Poisson(ParameterList const& p, Disc* d);
    ~Poisson();
    Disc* get_disc() { return disc; }
    apf::Field* get_soln() { return soln; }
    template <typename T>
    void build_resid(Evaluators& E);
    template <typename T>
    void build_functional(ParameterList const& params, Evaluators& E);
  private:
    void make_soln();
    ParameterList params;
    Disc* disc;
    apf::Field* soln;
};

Poisson* create_poisson(ParameterList const& p, Disc* d);
void destroy_poisson(Poisson* m);

}

#endif
