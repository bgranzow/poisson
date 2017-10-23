#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfShape.h>
#include <ma.h>

#include "goal_control.hpp"
#include "goal_nested.hpp"

namespace goal {


Nested::Nested(Disc* d, const int m) {
  double t0 = time();
  mode = m;
  is_base = false;
  sets = d->get_model_sets();
  base_mesh = d->get_apf_mesh();
  base_ve_nmbr = 0;
  nested_ve_nmbr = 0;
  nested_nmbr = 0;
  create_base_map();
  create_nested_mesh();
  initialize_nested_nmbr();
  refine_mesh();
  initialize();
  double t1 = time();
  print(" > nested mesh built in %f seconds", t1 - t0);
}

Nested::~Nested() {
  apf::destroyGlobalNumbering(nested_nmbr);
}

void Nested::set_adjoint(RCP<VectorT> z) {
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(nmbr, nodes);
  auto data = z->get1dView();
  auto zu = mesh->findField("zu");
  for (size_t n = 0; n < nodes.size(); ++n) {
    auto node = nodes[n];
    auto ent = node.entity;
    auto lnode = node.node;
    if (! mesh->isOwned(ent)) continue;
    GO row = get_gid(node, 0);
    LO lrow = owned_map->getLocalElement(row);
    double soln = data[lrow];
    apf::setScalar(zu, ent, lnode, soln);
  }
  apf::synchronize(zu);
}

void Nested::transfer_adjoint() {
  double zpress = 0.0;
  apf::Vector3 zdisp(0,0,0);
  auto P2 = apf::getSerendipity();
  auto nested_zu = mesh->findField("zu");
  auto nested_zp = mesh->findField("zp");
  auto zu_fine = apf::createField(base_mesh, "zu_fine", apf::VECTOR, P2);
  auto zp_fine = apf::createField(base_mesh, "zp_fine", apf::SCALAR, P2);
  auto zu = apf::createFieldOn(base_mesh, "zu", apf::VECTOR);
  auto zp = apf::createFieldOn(base_mesh, "zp", apf::SCALAR);
  apf::MeshEntity* vtx;
  apf::MeshIterator* vertices = mesh->begin(0);
  while ((vtx = mesh->iterate(vertices))) {
    auto nmbr = apf::getNumber(nested_nmbr, vtx, 0);
    auto base_ent = map[nmbr];
    apf::getVector(nested_zu, vtx, 0, zdisp);
    zpress = apf::getScalar(nested_zp, vtx, 0);
    apf::setVector(zu_fine, base_ent, 0, zdisp);
    apf::setScalar(zp_fine, base_ent, 0, zpress);
  }
  mesh->end(vertices);
  apf::projectField(zu, zu_fine);
  apf::projectField(zp, zp_fine);
}

void Nested::create_base_map() {
  auto s = apf::getSerendipity();
  base_ve_nmbr = apf::makeGlobal(
      apf::numberOwnedNodes(base_mesh, "nve", s));
  apf::synchronize(base_ve_nmbr);
  for (int dim = 0; dim <= 1; ++dim) {
    apf::MeshEntity* ent;
    apf::MeshIterator* it = base_mesh->begin(dim);
    while ((ent = base_mesh->iterate(it)))
      map[ apf::getNumber(base_ve_nmbr, ent, 0) ] = ent;
    base_mesh->end(it);
  }
}

void Nested::create_nested_mesh() {
  auto model = base_mesh->getModel();
  mesh = apf::createMdsMesh(model, base_mesh);
  apf::disownMdsModel(mesh);
  nested_ve_nmbr = mesh->findGlobalNumbering("nve_global");
  GOAL_DEBUG_ASSERT(nested_ve_nmbr);
  apf::destroyGlobalNumbering(base_ve_nmbr);
  base_ve_nmbr = 0;
}

void Nested::initialize_nested_nmbr() {
  auto s = mesh->getShape();
  nested_nmbr = createGlobalNumbering(mesh, "n", s);
  apf::MeshEntity* vtx;
  apf::MeshIterator* vertices = mesh->begin(0);
  while ((vtx = mesh->iterate(vertices))) {
    long gid = apf::getNumber(nested_ve_nmbr, vtx, 0);
    apf::number(nested_nmbr, vtx, 0, gid);
  }
  mesh->end(vertices);
}

class NmbrTransfer : public ma::SolutionTransfer {
  public:
    NmbrTransfer(
        apf::GlobalNumbering* ve_nmbr,
        apf::GlobalNumbering* nmbr);
    bool hasNodesOn(int dim);
    void onVertex(
        apf::MeshElement* parent,
        apf::Vector3 const& xi,
        apf::MeshEntity* vtx);
  private:
    apf::GlobalNumbering* ve_nmbr;
    apf::GlobalNumbering* nmbr;
};

NmbrTransfer::NmbrTransfer(
    apf::GlobalNumbering* ve,
    apf::GlobalNumbering* n) {
  ve_nmbr = ve;
  nmbr = n;
}

bool NmbrTransfer::hasNodesOn(int dim) {
  if (dim == 0) return true;
  else return false;
}

void NmbrTransfer::onVertex(
    apf::MeshElement* parent,
    apf::Vector3 const&,
    apf::MeshEntity* vtx) {
  auto ent = apf::getMeshEntity(parent);
  auto gid = apf::getNumber(ve_nmbr, ent, 0);
  apf::number(nmbr, vtx, 0, gid);
}

struct LongRefiner : public ma::IdentitySizeField {
  LongRefiner(apf::Mesh2* m);
  bool shouldSplit(apf::MeshEntity* edge);
  void indicate_edges();
  void gather_edges();
  void mark_edges();
  int dim;
  apf::Mesh2* mesh;
  apf::Field* marks;
  std::set<apf::MeshEntity*> stored;
};

LongRefiner::LongRefiner(apf::Mesh2* m) :
    ma::IdentitySizeField(m) {
  mesh = m;
  dim = mesh->getDimension();
  mark_edges();
}

bool LongRefiner::shouldSplit(apf::MeshEntity* edge) {
  if (stored.count(edge)) return true;
  else return false;
}

static apf::MeshEntity* find_max_edge(
    apf::Mesh* m, apf::MeshEntity* elem) {
  apf::MeshEntity* max = 0;
  apf::Downward edges;
  int nedges = m->getDownward(elem, 1, edges);
  for (int i = 0; i < nedges-1; ++i) {
    if (apf::measure(m, edges[i]) > apf::measure(m, edges[i+1]))
      max = edges[i];
    else
      max = edges[i+1];
  }
  return max;
}

void LongRefiner::indicate_edges() {
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = mesh->begin(dim);
  while ((elem = mesh->iterate(elems))) {
    auto edge = find_max_edge(mesh, elem);
    apf::setScalar(marks, edge, 0, 1.0);
  }
  mesh->end(elems);
  apf::accumulate(marks);
}

void LongRefiner::gather_edges() {
  apf::MeshEntity* edge;
  apf::MeshIterator* edges = mesh->begin(1);
  while ((edge = mesh->iterate(edges)))
    if (apf::getScalar(marks, edge, 0) > 0)
      stored.insert(edge);
  mesh->end(edges);
}

void LongRefiner::mark_edges() {
  auto s = apf::getConstant(1);
  marks = apf::createField(mesh, "mark", apf::SCALAR, s);
  apf::zeroField(marks);
  indicate_edges();
  gather_edges();
  apf::destroyField(marks);
}

struct SingleRefiner : public ma::IdentitySizeField {
  SingleRefiner(apf::Mesh2* m);
  bool shouldSplit(apf::MeshEntity* edge);
  void mark_edge(apf::MeshEntity* edge);
  bool needs_marking(apf::Adjacent& elems);
  void indicate_edges();
  void gather_edges();
  void mark_edges();
  int dim;
  apf::Mesh2* mesh;
  apf::Field* edge_mark;
  apf::Field* elem_mark;
  std::set<apf::MeshEntity*> stored;
};

SingleRefiner::SingleRefiner(apf::Mesh2* m) :
    ma::IdentitySizeField(m) {
  mesh = m;
  dim = mesh->getDimension();
  mark_edges();
}

bool SingleRefiner::shouldSplit(apf::MeshEntity* edge) {
  if (stored.count(edge)) return true;
  else return false;
}

void SingleRefiner::mark_edge(apf::MeshEntity* edge) {
  apf::setScalar(edge_mark, edge, 0, 1.0);
  apf::Adjacent elems;
  mesh->getAdjacent(edge, dim, elems);
  for (size_t i = 0; i < elems.getSize(); ++i)
    apf::setScalar(elem_mark, elems[i], 0, 1.0);
}

bool SingleRefiner::needs_marking(apf::Adjacent& elems) {
  bool mark = true;
  for (size_t i = 0; i < elems.getSize(); ++i)
    if (apf::getScalar(elem_mark, elems[i], 0) > 0)
      mark = false;
  return mark;
}

void SingleRefiner::indicate_edges() {
  apf::Adjacent elems;
  apf::MeshEntity* edge;
  apf::MeshIterator* edges = mesh->begin(1);
  while ((edge = mesh->iterate(edges))) {
    mesh->getAdjacent(edge, dim, elems);
    if (needs_marking(elems))
      mark_edge(edge);
  }
  mesh->end(edges);
  apf::accumulate(edge_mark);
}

void SingleRefiner::gather_edges() {
  apf::MeshEntity* edge;
  apf::MeshIterator* edges = mesh->begin(1);
  while ((edge = mesh->iterate(edges)))
    if (apf::getScalar(edge_mark, edge, 0) > 0)
      stored.insert(edge);
  mesh->end(edges);
}

void SingleRefiner::mark_edges() {
  apf::FieldShape* sedge = apf::getConstant(1);
  apf::FieldShape* selem = apf::getConstant(dim);
  edge_mark = apf::createField(mesh, "edge_mark", apf::SCALAR, sedge);
  elem_mark = apf::createField(mesh, "elem_mark", apf::SCALAR, selem);
  apf::zeroField(edge_mark);
  apf::zeroField(elem_mark);
  indicate_edges();
  gather_edges();
  apf::destroyField(edge_mark);
  apf::destroyField(elem_mark);
}

void Nested::refine_uniform() {
  goal::print(" > nested: full");
  ma::AutoSolutionTransfer trans(mesh);
  auto nt = new NmbrTransfer(nested_ve_nmbr, nested_nmbr);
  trans.add(nt);
  auto in = ma::configureUniformRefine(mesh, 1, &trans);
  in->shouldFixShape = false;
  in->shouldSnap = false;
  ma::adapt(in);
}

void Nested::refine_long() {
  goal::print(" > nested: long");
  auto size = new LongRefiner(mesh);
  auto in = ma::configureIdentity(mesh, size);
  in->shouldFixShape = false;
  in->shouldSnap = false;
  in->maximumIterations = 1;
  ma::adapt(in);
  delete size;
}

void Nested::refine_single() {
  goal::print(" > nested: single");
  auto size = new SingleRefiner(mesh);
  auto in = ma::configureIdentity(mesh, size);
  in->shouldFixShape = false;
  in->shouldSnap = false;
  in->maximumIterations = 1;
  ma::adapt(in);
  delete size;
}

void Nested::refine_mesh() {
  if (mode == FULL) refine_uniform();
  else if (mode == LONG) refine_long();
  else if (mode == SINGLE) refine_single();
  else fail("unknown refine mode: %d", mode);
  apf::destroyGlobalNumbering(nested_ve_nmbr);
  apf::reorderMdsMesh(mesh);
  mesh->verify();
  nested_ve_nmbr = 0;
}

Nested* create_nested(Disc* d, const int mode) {
  return new Nested(d, mode);
}

void destroy_nested(Nested* n) {
  delete n;
}

}
