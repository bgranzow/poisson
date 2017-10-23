#include <apf.h>
#include <apfMesh2.h>
#include <apfNumbering.h>

#include "goal_control.hpp"
#include "goal_dbcs.hpp"
#include "goal_disc.hpp"
#include "goal_sol_info.hpp"

namespace goal {

using Teuchos::Array;
using Teuchos::getValue;

static void validate_params(ParameterList const& p, SolInfo* s) {
  auto d = s->get_disc();
  for (auto it = p.begin(); it != p.end(); ++it) {
    auto entry = p.entry(it);
    auto a = getValue<Array<std::string>>(entry);
    GOAL_DEBUG_ASSERT(a.size() == 2);
    auto set = a[0];
    auto nodes = d->get_nodes(set);
  }
}

static double get_val(
    apf::Field* f,
    std::string const& val,
    apf::Node const& node,
    const double t) {
  apf::Vector3 x;
  auto e = node.entity;
  auto m = apf::getMesh(f);
  m->getPoint(e, 0, x);
  double v = eval(val, x[0], x[1], x[2], t);
  return v;
}

void set_resid_dbcs(ParameterList const& p, SolInfo* s, const double t) {
  validate_params(p, s);
  auto d = s->get_disc();
  auto R = s->owned->R;
  auto u = d->get_apf_mesh()->findField("u");
  for (auto it = p.begin(); it != p.end(); ++it) {
    auto entry = p.entry(it);
    auto a = getValue<Array<std::string>>(entry);
    auto set = a[0];
    auto val = a[1];
    auto nodes = d->get_nodes(set);
    for (size_t node = 0; node < nodes.size(); ++node) {
      auto n = nodes[node];
      GO row = d->get_gid(n, 0);
      auto sol = apf::getScalar(u, n.entity, n.node);
      double v = get_val(u, val, n, t);
      R->replaceGlobalValue(row, sol - v);
    }
  }
}

void set_jac_dbcs(ParameterList const& p, SolInfo* s, const double t) {
  validate_params(p, s);
  auto d = s->get_disc();
  auto R = s->owned->R;
  auto dRdu = s->owned->dRdu;
  auto dMdu = s->owned->dMdu;
  Array<ST> entries, entry(1);
  Array<GO> indices, index(1);
  entry[0] = 1.0;
  auto u = d->get_apf_mesh()->findField("u");
  for (auto it = p.begin(); it != p.end(); ++it) {
    auto pentry = p.entry(it);
    auto a = getValue<Array<std::string>>(pentry);
    auto set = a[0];
    auto val = a[1];
    auto nodes = d->get_nodes(set);
    for (size_t node = 0; node < nodes.size(); ++node) {
      auto n = nodes[node];
      GO row = d->get_gid(n, 0);
      index[0] = row;
      auto sol = apf::getScalar(u, n.entity, n.node);
      double v = get_val(u, val, n, t);
      R->replaceGlobalValue(row, sol - v);
      dMdu->replaceGlobalValue(row, 0.0);
      size_t num_cols = dRdu->getNumEntriesInGlobalRow(row);
      indices.resize(num_cols);
      entries.resize(num_cols);
      dRdu->getGlobalRowCopy(row, indices(), entries(), num_cols);
      for (size_t col = 0; col < num_cols; ++col) entries[col] = 0.0;
      dRdu->replaceGlobalValues(row, indices(), entries());
      dRdu->replaceGlobalValues(row, index(), entry());
    }
  }
}

}
