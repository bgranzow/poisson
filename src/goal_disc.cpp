#include <apf.h>
#include <apfAlbany.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfShape.h>
#include <gmi_mesh.h>
#include <Teuchos_ParameterList.hpp>

#ifdef GOAL_ENABLE_SNAPPING
#include <gmi_sim.h>
#include <SimUtil.h>
#endif

#include "goal_control.hpp"
#include "goal_disc.hpp"

namespace goal {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("geom file", "");
  p.set<std::string>("mesh file", "");
  p.set<std::string>("assoc file", "");
  return p;
}

static apf::StkModels* read_sets(apf::Mesh* m, ParameterList const& p) {
  auto sets = new apf::StkModels;
  auto fn = p.get<std::string>("assoc file");
  auto filename = fn.c_str();
  print("reading association file: %s", filename);
  static std::string const setNames[3] = {
    "node set", "side set", "elem set"};
  auto d = m->getDimension();
  int dims[3] = {0, d - 1, d};
  std::ifstream f(filename);
  if (!f.good()) fail("cannot open file: %s", filename);
  std::string sline;
  int lc = 0;
  while (std::getline(f, sline)) {
    if (!sline.length()) break;
    ++lc;
    int sdi = -1;
    for (int di = 0; di < 3; ++di)
      if (sline.compare(0, setNames[di].length(), setNames[di]) == 0) sdi = di;
    if (sdi == -1)
      fail("invalid association line # %d:\n\t%s", lc, sline.c_str());
    int sd = dims[sdi];
    std::stringstream strs(sline.substr(setNames[sdi].length()));
    auto set = new apf::StkModel();
    strs >> set->stkName;
    int nents;
    strs >> nents;
    if (!strs) fail("invalid association line # %d:\n\t%s", lc, sline.c_str());
    for (int ei = 0; ei < nents; ++ei) {
      std::string eline;
      std::getline(f, eline);
      if (!f || !eline.length())
        fail("invalid association after line # %d", lc);
      ++lc;
      std::stringstream strs2(eline);
      int mdim, mtag;
      strs2 >> mdim >> mtag;
      if (!strs2) fail("bad associations line # %d:\n\t%s", lc, eline.c_str());
      set->ents.push_back(m->findModelEntity(mdim, mtag));
      if (!set->ents.back())
        fail("no model entity with dim: %d and tag: %d", mdim, mtag);
    }
    sets->models[sd].push_back(set);
  }
  sets->computeInverse();
  return sets;
}

static void initialize_sim() {
#ifdef GOAL_ENABLE_SNAPPING
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();
#endif
}

static void finalize_sim() {
#ifdef GOAL_ENABLE_SNAPPING
  gmi_sim_stop();
  Sim_unregisterAllKeys();
#endif
}

static void load_mesh(apf::Mesh2** mesh, ParameterList const& p) {
  gmi_register_mesh();
  initialize_sim();
  auto geom_file = p.get<std::string>("geom file");
  auto mesh_file = p.get<std::string>("mesh file");
  auto g = geom_file.c_str();
  auto m = mesh_file.c_str();
  *mesh = apf::loadMdsMesh(g, m);
}

Disc::Disc() {
}

Disc::Disc(ParameterList const& p) {
  p.validateParameters(get_valid_params(), 0);
  is_base = true;
  load_mesh(&mesh, p);
  sets = read_sets(mesh, p);
  apf::reorderMdsMesh(mesh);
  mesh->verify();
  initialize();
}

Disc::~Disc() {
  destroy_data();
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  if (is_base) delete sets;
  if (is_base) finalize_sim();
}

std::string Disc::get_elem_set_name(const int es_idx) const {
  GOAL_DEBUG_ASSERT(es_idx < num_elem_sets);
  return sets->models[num_dims][es_idx]->stkName;
}

std::string Disc::get_side_set_name(const int ss_idx) const {
  GOAL_DEBUG_ASSERT(ss_idx < num_side_sets);
  return sets->models[num_dims-1][ss_idx]->stkName;
}

std::string Disc::get_node_set_name(const int ns_idx) const {
  GOAL_DEBUG_ASSERT(ns_idx < num_node_sets);
  return sets->models[0][ns_idx]->stkName;
}

ElemSet const& Disc::get_elems(std::string const& es_name) {
  GOAL_DEBUG_ASSERT(elem_sets.count(es_name));
  return elem_sets[es_name];
}

SideSet const& Disc::get_sides(std::string const& ss_name) {
  GOAL_DEBUG_ASSERT(side_sets.count(ss_name));
  return side_sets[ss_name];
}

NodeSet const& Disc::get_nodes(std::string const& ns_name) {
  GOAL_DEBUG_ASSERT(node_sets.count(ns_name));
  return node_sets[ns_name];
}

int Disc::get_num_nodes(apf::MeshEntity* e) {
  auto type = mesh->getType(e);
  auto shape = mesh->getShape();
  auto es = shape->getEntityShape(type);
  return es->countNodes();
}

int Disc::get_num_dofs(apf::MeshEntity* e) {
  return get_num_nodes(e) * num_eqs;
}

static LO get_dof(const LO nid, const int eq, const int neq) {
  return nid*neq + eq;
}

static GO get_gdof(const GO nid, const int eq, const int neq) {
  return nid*neq + eq;
}

GO Disc::get_gid(apf::MeshEntity* e, const int n, const int eq) {
  apf::NewArray<long> node_ids;
  apf::getElementNumbers(nmbr, e, node_ids);
  return get_gdof(node_ids[n], eq, num_eqs);
}

GO Disc::get_gid(apf::Node const& n, const int eq) {
  GO nid = apf::getNumber(nmbr, n.entity, n.node, 0);
  return get_gdof(nid, eq, num_eqs);
}

void Disc::get_gids(apf::MeshEntity* e, std::vector<GO>& gids) {
  int dof = 0;
  apf::NewArray<long> node_ids;
  gids.resize(get_num_dofs(e));
  apf::getElementNumbers(nmbr, e, node_ids);
  for (int n = 0; n < get_num_nodes(e); ++n)
  for (int eq = 0; eq < num_eqs; ++eq)
    gids[dof++] = get_gdof(node_ids[n], eq, num_eqs);
}

void Disc::build_data() {
  auto t0 = time();
  compute_owned_maps();
  compute_coords();
  compute_ghost_map();
  compute_graphs();
  compute_elem_sets();
  compute_side_sets();
  compute_node_sets();
  auto t1 = time();
  print(" > disc: data built in %f seconds", t1 - t0);
}

void Disc::destroy_data() {
  if (nmbr) apf::destroyGlobalNumbering(nmbr);
  for (int i = 0; i < get_num_elem_sets(); ++i)
    elem_sets[get_elem_set_name(i)].resize(0);
  for (int i = 0; i < get_num_side_sets(); ++i)
    side_sets[get_side_set_name(i)].resize(0);
  for (int i = 0; i < get_num_node_sets(); ++i)
    node_sets[get_node_set_name(i)].resize(0);
  node_map = Teuchos::null;
  owned_map = Teuchos::null;
  ghost_map = Teuchos::null;
  coords = Teuchos::null;
  owned_graph = Teuchos::null;
  ghost_graph = Teuchos::null;
  nmbr = 0;
}

void Disc::add_soln(RCP<VectorT> du) {
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(nmbr, nodes);
  auto data = du->get1dView();
  auto u = mesh->findField("u");
  for (size_t n = 0; n < nodes.size(); ++n) {
    auto node = nodes[n];
    auto ent = node.entity;
    auto lnode = node.node;
    if (! mesh->isOwned(ent)) continue;
    double soln = apf::getScalar(u, ent, lnode);
    GO row = get_gid(node, 0);
    LO lrow = owned_map->getLocalElement(row);
    soln += data[lrow];
    apf::setScalar(u, ent, lnode, soln);
  }
  apf::synchronize(u);
}

void Disc::initialize() {
  num_dims = mesh->getDimension();
  num_eqs = 1;
  num_elem_sets = sets->models[num_dims].size();
  num_side_sets = sets->models[num_dims-1].size();
  num_node_sets = sets->models[0].size();
  comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  nmbr = 0;
}

void Disc::compute_owned_maps() {
  GOAL_DEBUG_ASSERT(! nmbr);
  nmbr = apf::makeGlobal(apf::numberOwnedNodes(mesh, "nmbr"));
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(nmbr, owned);
  auto num_owned = owned.getSize();
  Teuchos::Array<GO> indices(num_owned);
  for (size_t n = 0; n < num_owned; ++n)
    indices[n] = apf::getNumber(nmbr, owned[n]);
  node_map = Tpetra::createNonContigMap<LO, GO>(indices, comm);
  indices.resize(num_eqs * num_owned);
  for (size_t n = 0; n < num_owned; ++n) {
    GO gid = apf::getNumber(nmbr, owned[n]);
    for (int eq = 0; eq < num_eqs; ++eq)
      indices[get_dof(n, eq, num_eqs)] = get_gdof(gid, eq, num_eqs);
  }
  owned_map = Tpetra::createNonContigMap<LO, GO>(indices, comm);
}

void Disc::compute_coords() {
  coords = rcp(new MultiVectorT(node_map, num_dims, false));
  apf::Vector3 x(0, 0, 0);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    auto node = owned[n];
    mesh->getPoint(node.entity, node.node, x);
    for (int dim = 0; dim < num_dims; ++dim)
      coords->replaceLocalValue(n, dim, x[dim]);
  }
}

void Disc::compute_ghost_map() {
  apf::synchronize(nmbr);
  apf::DynamicArray<apf::Node> ghost;
  apf::getNodes(nmbr, ghost);
  auto num_ghost = ghost.getSize();
  Teuchos::Array<GO> indices(num_eqs * num_ghost);
  for (size_t n = 0; n < num_ghost; ++n) {
    GO gid = apf::getNumber(nmbr, ghost[n]);
    for (int eq = 0; eq < num_eqs; ++eq)
      indices[get_dof(n, eq, num_eqs)] = get_gdof(gid, eq, num_eqs);
  }
  ghost_map = Tpetra::createNonContigMap<LO, GO>(indices, comm);
}

void Disc::compute_graphs() {
  int est = 300;
  owned_graph = rcp(new GraphT(owned_map, est));
  ghost_graph = rcp(new GraphT(ghost_map, est));
  apf::MeshEntity* elem;
  auto elems = mesh->begin(num_dims);
  while ((elem = mesh->iterate(elems))) {
    apf::NewArray<long> gids;
    int nnodes = apf::getElementNumbers(nmbr, elem, gids);
    for (int i = 0; i < nnodes; ++i) {
      for (int j = 0; j < num_eqs; ++j) {
        GO row = get_gdof(gids[i], j, num_eqs);
        for (int k = 0; k < nnodes; ++k) {
          for (int l = 0; l < num_eqs; ++l) {
            GO col = get_gdof(gids[k], l, num_eqs);
            auto col_av = Teuchos::arrayView(&col, 1);
            ghost_graph->insertGlobalIndices(row, col_av);
  }}}}}
  mesh->end(elems);
  ghost_graph->fillComplete();
  auto exporter = rcp(new ExportT(ghost_map, owned_map));
  owned_graph->doExport(*ghost_graph, *exporter, Tpetra::INSERT);
  owned_graph->fillComplete();
}

void Disc::compute_elem_sets() {
  for (int i = 0; i < num_elem_sets; ++i)
    elem_sets[ get_elem_set_name(i) ].resize(0);
  apf::MeshEntity* elem;
  auto it = mesh->begin(num_dims);
  while ((elem = mesh->iterate(it))) {
    auto mr = mesh->toModel(elem);
    auto stkm = sets->invMaps[num_dims][mr];
    auto name = stkm->stkName;
    elem_sets[name].push_back(elem);
  }
  mesh->end(it);
}

void Disc::compute_side_sets() {
  for (int i = 0; i < num_side_sets; ++i)
    side_sets[ get_side_set_name(i) ].resize(0);
  apf::MeshEntity* side;
  auto it = mesh->begin(num_dims - 1);
  while ((side = mesh->iterate(it))) {
    auto me = mesh->toModel(side);
    if (!sets->invMaps[num_dims - 1].count(me)) continue;
    auto stkm = sets->invMaps[num_dims -1][me];
    auto name = stkm->stkName;
    apf::Up adj_elems;
    mesh->getUp(side, adj_elems);
    GOAL_DEBUG_ASSERT_VERBOSE(adj_elems.n == 1,
        "side set defined on non-manifold geometric entity?");
    side_sets[name].push_back(side);
  }
  mesh->end(it);
}

void Disc::compute_node_sets() {
  for (int i = 0; i < num_node_sets; ++i)
    node_sets[ get_node_set_name(i) ].resize(0);
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(nmbr, nodes);
  for (size_t n = 0; n < nodes.size(); ++n) {
    auto node = nodes[n];
    auto ent = node.entity;
    if (! mesh->isOwned(ent)) continue;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(
        mesh, sets->invMaps[0], mesh->toModel(ent), mset);
    if (mset.empty()) continue;
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      auto ns = *mit;
      auto nsn = ns->stkName;
      node_sets[nsn].push_back(node);
    }
  }
}

Disc* create_disc(ParameterList const& p) {
  return new Disc(p);
}

void destroy_disc(Disc* d) {
  delete d;
}

}
