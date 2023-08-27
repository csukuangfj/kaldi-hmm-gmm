// kaldi-hmm-gmm/csrc/tree-renderer.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/tree/tree-renderer.cc
#include "kaldi-hmm-gmm/csrc/tree-renderer.h"

#include <string>
#include <utility>

#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/kaldi-utils.h"
#include "kaldi_native_io/csrc/stl-utils.h"

namespace khg {

const int32_t TreeRenderer::kEdgeWidth = 1;
const int32_t TreeRenderer::kEdgeWidthQuery = 3;
const char *TreeRenderer::kEdgeColor = "black";
const char *TreeRenderer::kEdgeColorQuery = "red";

void TreeRenderer::RenderNonLeaf(int32_t id, const EventKeyType &key,
                                 bool in_query) {
  std::string color = in_query ? kEdgeColorQuery : kEdgeColor;
  int32_t width = in_query ? kEdgeWidthQuery : kEdgeWidth;
  std::string label;
  if (key == kPdfClass) {
    label = "\"PdfClass = ?\"";
  } else if (key == 0) {
    if (N_ == 1 && P_ == 0)  // monophone tree?
      label = "\"Phone = ?\"";
    else if (N_ == 3 && P_ == 1)  // triphone tree?
      label = "\"LContext = ?\"";
  } else if (key == 2 && N_ == 3 && P_ == 1) {
    label = "\"RContext = ?\"";
  } else if (key >= 0 && key <= N_ - 1) {
    if (P_ == key) {
      label = "\"Center = ?\"";
    } else {
      std::ostringstream oss;
      oss << "\"Ctx Position " << key << " = ?\"";
      label = oss.str();
    }
  } else {
    KHG_ERR << "Invalid decision tree key: " << key;
  }

  out_ << id << "[label=" << label << ", color=" << color
       << ", penwidth=" << width << "];" << std::endl;
}

std::string TreeRenderer::MakeEdgeLabel(
    const EventKeyType &key, const ConstIntegerSet<EventValueType> &intset) {
  std::ostringstream oss;
  ConstIntegerSet<EventValueType>::iterator child = intset.begin();
  for (; child != intset.end(); ++child) {
    if (child != intset.begin()) oss << ", ";
    if (key != kPdfClass) {
      std::string phone = phone_syms_.Find(static_cast<int64_t>(*child));
      if (phone.empty()) KHG_ERR << "No phone found for Phone ID " << *child;
      oss << phone;
    } else {
      oss << *child;
    }
  }

  return oss.str();
}

void TreeRenderer::RenderSplit(const EventType *query, int32_t id) {
  kaldiio::ExpectToken(is_, binary_, "SE");
  EventKeyType key;
  kaldiio::ReadBasicType(is_, binary_, &key);
  ConstIntegerSet<EventValueType> yes_set;
  yes_set.Read(is_, binary_);
  kaldiio::ExpectToken(is_, binary_, "{");

  EventValueType value = -30000000;  // just a value I guess is invalid
  if (query != NULL) EventMap::Lookup(*query, key, &value);
  const EventType *query_yes = yes_set.count(value) ? query : NULL;
  const EventType *query_no = (query_yes == NULL) ? query : NULL;
  std::string color_yes = (query_yes) ? kEdgeColorQuery : kEdgeColor;
  std::string color_no = (query && !query_yes) ? kEdgeColorQuery : kEdgeColor;
  int32_t width_yes = (query_yes) ? kEdgeWidthQuery : kEdgeWidth;
  int32_t width_no = (query && !query_yes) ? kEdgeWidthQuery : kEdgeWidth;
  RenderNonLeaf(id, key, (query != NULL));  // Draw the node itself
  std::string yes_label = MakeEdgeLabel(key, yes_set);
  out_ << "\t" << id << " -> " << next_id_++ << " [";  // YES edge
  if (use_tooltips_) {
    out_ << "tooltip=\"" << yes_label << "\", label=YES"
         << ", penwidth=" << width_yes << ", color=" << color_yes << "];\n";
  } else {
    out_ << "label=\"" << yes_label << "\", penwidth=" << width_yes
         << ", penwidth=" << width_yes << ", color=" << color_yes << "];\n";
  }
  RenderSubTree(query_yes, next_id_ - 1);  // Render YES subtree
  out_ << "\t" << id << " -> " << next_id_++ << "[label=NO"  // NO edge
       << ", color=" << color_no << ", penwidth=" << width_no << "];\n";
  RenderSubTree(query_no, next_id_ - 1);  // Render NO subtree

  kaldiio::ExpectToken(is_, binary_, "}");
}

void TreeRenderer::RenderTable(const EventType *query, int32_t id) {
  kaldiio::ExpectToken(is_, binary_, "TE");
  EventKeyType key;
  kaldiio::ReadBasicType(is_, binary_, &key);
  uint32 size;
  kaldiio::ReadBasicType(is_, binary_, &size);
  kaldiio::ExpectToken(is_, binary_, "(");

  EventValueType value = -3000000;  // just a value I hope is invalid
  if (query != NULL) EventMap::Lookup(*query, key, &value);
  RenderNonLeaf(id, key, (query != NULL));
  for (size_t t = 0; t < size; t++) {
    std::string color = (t == value) ? kEdgeColorQuery : kEdgeColor;
    int32_t width = (t == value) ? kEdgeWidthQuery : kEdgeWidth;
    std::ostringstream label;
    if (key == kPdfClass) {
      label << t;
    } else if (key >= 0 && key < N_) {
      if (t == 0 || kaldiio::PeekToken(is_, binary_) == 'N') {
        kaldiio::ExpectToken(is_, binary_,
                             "NULL");  // consume the invalid/NULL entry
        continue;
      }
      std::string phone = phone_syms_.Find(static_cast<int64_t>(t));
      if (phone.empty())
        KHG_ERR << "Phone ID found in a TableEventMap, but not in the "
                << "phone symbol table! ID: " << t;
      label << phone;
    } else {
      KHG_ERR << "TableEventMap: Invalid event key: " << key;
    }
    // draw the edge to the child subtree
    out_ << "\t" << id << " -> " << next_id_++ << " [label=" << label.str()
         << ", color=" << color << ", penwidth=" << width << "];\n";
    const EventType *query_child = (t == value) ? query : NULL;
    RenderSubTree(query_child, next_id_ - 1);  // render the child subtree
  }

  kaldiio::ExpectToken(is_, binary_, ")");
}

void TreeRenderer::RenderConstant(const EventType *query, int32_t id) {
  kaldiio::ExpectToken(is_, binary_, "CE");
  EventAnswerType answer;
  kaldiio::ReadBasicType(is_, binary_, &answer);

  std::string color = (query != NULL) ? kEdgeColorQuery : kEdgeColor;
  int32_t width = (query != NULL) ? kEdgeWidthQuery : kEdgeWidth;
  out_ << id << "[shape=doublecircle, label=" << answer << ",color=" << color
       << ", penwidth=" << width << "];\n";
}

void TreeRenderer::RenderSubTree(const EventType *query, int32_t id) {
  char c = kaldiio::Peek(is_, binary_);
  if (c == 'N') {
    kaldiio::ExpectToken(is_, binary_, "NULL");  // consume NULL entries
    return;
  } else if (c == 'C') {
    RenderConstant(query, id);
  } else if (c == 'T') {
    RenderTable(query, id);
  } else if (c == 'S') {
    RenderSplit(query, id);
  } else {
    KHG_ERR << "EventMap::read, was not expecting character "
            << kaldiio::CharToString(c) << ", at file position " << is_.tellg();
  }
}

void TreeRenderer::Render(const EventType *query = 0) {
  kaldiio::ExpectToken(is_, binary_, "ContextDependency");
  kaldiio::ReadBasicType(is_, binary_, &N_);
  kaldiio::ReadBasicType(is_, binary_, &P_);
  kaldiio::ExpectToken(is_, binary_, "ToPdf");
  if (query && query->size() != N_ + 1)
    KHG_ERR << "Invalid query size \"" << query->size() << "\"! Expected \""
            << N_ + 1 << '"';
  out_ << "digraph EventMap {\n";
  RenderSubTree(query, next_id_++);
  out_ << "}\n";
  kaldiio::ExpectToken(is_, binary_, "EndContextDependency");
}
}  // namespace khg
