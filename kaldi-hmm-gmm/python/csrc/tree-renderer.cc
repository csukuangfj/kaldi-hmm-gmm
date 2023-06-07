// kaldi-hmm-gmm/python/csrc/tree-renderer.cc
//
// Copyright (c)  2022  Xiaomi Corporation
//
// this file is copied and modified from
// kaldi/src/bin/draw-tree.cc

#include "kaldi-hmm-gmm/python/csrc/tree-renderer.h"

#include <string>
#include <utility>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/tree-renderer.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

static void MakeEvent(const std::string &qry, fst::SymbolTable *phone_syms,
                      EventType **query) {
  EventType *query_event = new EventType();
  size_t found, old_found = 0;
  EventKeyType key = kPdfClass;  // this code in fact relies on kPdfClass = -1
  while ((found = qry.find('/', old_found)) != std::string::npos) {
    std::string valstr = qry.substr(old_found, found - old_found);
    EventValueType value;
    if (key == kPdfClass) {
      value = static_cast<EventValueType>(atoi(valstr.c_str()));
      if (value < 0) {  // not valid pdf-class
        KHG_ERR << "Bad query: invalid pdf-class (" << valstr << ')';
      }
    } else {
      value = static_cast<EventValueType>(phone_syms->Find(valstr.c_str()));
      if (value == -1) {  // fst::kNoSymbol
        KHG_ERR << "Bad query: invalid symbol (" << valstr << ')';
      }
    }
    query_event->push_back(std::make_pair(key++, value));
    old_found = found + 1;
  }
  std::string valstr = qry.substr(old_found);
  EventValueType value =
      static_cast<EventValueType>(phone_syms->Find(valstr.c_str()));
  if (value == -1) {  // fst::kNoSymbol
    KHG_ERR << "Bad query: invalid symbol (" << valstr << ')';
  }
  query_event->push_back(std::make_pair(key, value));

  *query = query_event;
}

static std::string DrawTree(const std::string &phones_txt,
                            const std::string &tree, bool use_tooltips = false,
                            const std::string &qry = "") {
  fst::SymbolTable *phones_symtab = nullptr;
  {
    std::ifstream is(phones_txt.c_str());
    phones_symtab = ::fst::SymbolTable::ReadText(is, phones_txt);
    if (!phones_symtab)
      KHG_ERR << "Could not read phones symbol table file " << phones_txt;
  }

  EventType *query = nullptr;
  if (!qry.empty()) MakeEvent(qry, phones_symtab, &query);

  std::ostringstream os;
  TreeRenderer *renderer = nullptr;
  {
    bool binary;
    kaldiio::Input ki(tree, &binary);
    renderer =
        new TreeRenderer(ki.Stream(), binary, os, *phones_symtab, use_tooltips);
    renderer->Render(query);
  }

  delete renderer;
  delete query;
  delete phones_symtab;

  return os.str();
}

void PybinTreeRenderer(py::module *m) {
  m->def("draw_tree", &DrawTree, py::arg("phones_txt"), py::arg("tree"),
         py::arg("use_tooltips") = false, py::arg("query") = "");
  // query is of the form:
  // pdf_class/context-phone-sym-1/context-phone-sym-2/context-phone-sym-N
}

}  // namespace khg
