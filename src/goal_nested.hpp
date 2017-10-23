#ifndef goal_nested_hpp
#define goal_nested_hpp

#include "goal_disc.hpp"

namespace goal {

enum RefineMode { FULL, LONG, SINGLE };

class Nested : public Disc {
  public:
    Nested(Disc* d, const int mode);
    ~Nested();
    void set_adjoint(RCP<VectorT> z);
    void transfer_adjoint();
  private:
    void create_base_map();
    void create_nested_mesh();
    void initialize_nested_nmbr();
    void refine_mesh();
    void refine_uniform();
    void refine_long();
    void refine_single();
    int mode;
    apf::Mesh2* base_mesh;
    apf::GlobalNumbering* base_ve_nmbr;
    apf::GlobalNumbering* nested_ve_nmbr;
    apf::GlobalNumbering* nested_nmbr;
    std::map<GO, apf::MeshEntity*> map;
};

Nested* create_nested(Disc* d, const int mode);
void destroy_nested(Nested* n);

}

#endif
