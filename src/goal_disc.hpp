#ifndef goal_disc_hpp
#define goal_disc_hpp

#include "goal_data_types.hpp"

namespace apf {
struct Node;
struct StkModels;
class Mesh2;
class MeshEntity;
template <class T> class NumberingOf;
typedef NumberingOf<long> GlobalNumbering;
}

namespace goal {

using Teuchos::RCP;
using Teuchos::ParameterList;
using ElemSet = std::vector<apf::MeshEntity*>;
using SideSet = std::vector<apf::MeshEntity*>;
using NodeSet = std::vector<apf::Node>;
using ElemSets = std::map<std::string, ElemSet>;
using SideSets = std::map<std::string, SideSet>;
using NodeSets = std::map<std::string, NodeSet>;

class Disc {
  public:
    Disc();
    Disc(ParameterList const& p);
    ~Disc();
    apf::Mesh2* get_apf_mesh() { return mesh; }
    apf::StkModels* get_model_sets() { return sets; }
    bool is_parent() const { return is_base; }
    int get_num_eqs() const { return num_eqs; }
    int get_num_dims() const { return num_dims; }
    int get_num_elem_sets() const { return num_elem_sets; }
    int get_num_side_sets() const { return num_side_sets; }
    int get_num_node_sets() const { return num_node_sets; }
    RCP<const MapT> get_owned_map() { return owned_map; }
    RCP<const MapT> get_ghost_map() { return ghost_map; }
    RCP<const GraphT> get_owned_graph() { return owned_graph; }
    RCP<const GraphT> get_ghost_graph() { return ghost_graph; }
    RCP<MultiVectorT> get_coords() { return coords; }
    std::string get_elem_set_name(const int es_idx) const;
    std::string get_side_set_name(const int es_idx) const;
    std::string get_node_set_name(const int es_idx) const;
    ElemSet const& get_elems(std::string const& es_name);
    SideSet const& get_sides(std::string const& ss_name);
    NodeSet const& get_nodes(std::string const& ns_name);
    int get_num_nodes(apf::MeshEntity* e);
    int get_num_dofs(apf::MeshEntity* e);
    GO get_gid(apf::MeshEntity* e, const int n, const int eq);
    GO get_gid(apf::Node const& n, const int eq);
    void get_gids(apf::MeshEntity* e, std::vector<GO>& gids);
    void add_soln(RCP<VectorT> du);
    void build_data();
    void destroy_data();
  protected:
    void initialize();
    void compute_owned_maps();
    void compute_coords();
    void compute_ghost_map();
    void compute_graphs();
    void compute_elem_sets();
    void compute_side_sets();
    void compute_node_sets();
    bool is_base;
    int num_dims;
    int num_eqs;
    int num_elem_sets;
    int num_side_sets;
    int num_node_sets;
    apf::Mesh2* mesh;
    apf::StkModels* sets;
    apf::GlobalNumbering* nmbr;
    ElemSets elem_sets;
    SideSets side_sets;
    NodeSets node_sets;
    RCP<const Comm> comm;
    RCP<const MapT> node_map;
    RCP<const MapT> owned_map;
    RCP<const MapT> ghost_map;
    RCP<MultiVectorT> coords;
    RCP<GraphT> owned_graph;
    RCP<GraphT> ghost_graph;
};

Disc* create_disc(ParameterList const& p);
void destroy_disc(Disc* d);

}

#endif
