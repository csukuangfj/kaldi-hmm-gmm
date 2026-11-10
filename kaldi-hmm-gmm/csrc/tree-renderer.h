// kaldi-hmm-gmm/csrc/tree-renderer.h
//
// Copyright 2012 Vassil Panayotov
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TREE_RENDERER_H_
#define KALDI_HMM_GMM_CSRC_TREE_RENDERER_H_

// This file is copied/modified from kaldi/src/tree/tree-renderer.h

#include <ostream>
#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/event-map.h"

/**
 * \file tree-renderer.h
 * \brief Renders a decision tree (\c EventMap) in GraphViz (dot) format.
 *
 * This file is copied/modified from kaldi/src/tree/tree-renderer.h.
 *
 * It declares \c TreeRenderer, which reads a decision tree (a
 * \c ContextDependency to-pdf \c EventMap) from a stream and writes its
 * description in GraphViz format, with phone-symbol edge labels taken from an
 * fst::SymbolTable.  It is used to visualize context-dependency trees, e.g.
 * by the Python \c draw_tree helper.
 */

namespace khg {

/**
 * \brief Parses a decision tree file and outputs its description in GraphViz
 *        format.
 *
 * The renderer walks the \c EventMap nodes (see event-map.h) and emits one
 * GraphViz node per tree node: leaf (\c ConstantEventMap) nodes as boxes,
 * and non-leaf (\c SplitEventMap, \c TableEventMap) nodes as ellipses
 * labeled with the question asked (a phone from the context window or the
 * pdf-class).  An optional query event can be rendered as a distinctly
 * colored trace through the tree.
 */
class TreeRenderer {
 public:
  // normal width of the edges and state contours
  static const int32_t kEdgeWidth;
  static const int32_t kEdgeWidthQuery;  // edge and state width when in query
  static const char *kEdgeColor;         // normal color for states and edges
  static const char *kEdgeColorQuery;    // edge and state color when in query

  /**
   * \brief Constructor.
   *
   * \param is            The stream from which the tree is read.
   * \param binary        Whether the input stream is binary.
   * \param os            The stream the GraphViz representation is written to.
   * \param phone_syms    Phone symbols to be used as edge labels.
   * \param use_tooltips  If true, use tooltips (useful in e.g. SVG) instead
   *                      of labels.
   */
  TreeRenderer(std::istream &is, bool binary,
               std::ostream &os,              // NOLINT
               fst::SymbolTable &phone_syms,  // NOLINT
               bool use_tooltips)
      : phone_syms_(phone_syms),
        is_(is),
        out_(os),
        binary_(binary),
        N_(-1),
        use_tooltips_(use_tooltips),
        next_id_(0) {}

  /**
   * \brief Renders the tree and, optionally, one event's path through it.
   *
   * Renders the tree and if the "query" parameter is not NULL
   * a distinctly colored trace corresponding to the event.
   *
   * \param query  If non-NULL, the event to trace through the tree (rendered
   *               distinctly colored); if NULL, the whole tree is rendered
   *               normally.
   */
  void Render(const EventType *query);

 private:
  /// Looks-up the next token from the stream and invokes
  /// the appropriate render method to visualize it
  void RenderSubTree(const EventType *query, int32_t id);

  /// Renders a leaf node (constant event map)
  void RenderConstant(const EventType *query, int32_t id);

  /// Renders a split event map node and the edges to the nodes
  /// representing YES and NO sets
  void RenderSplit(const EventType *query, int32_t id);

  /// Renders a table event map node and the edges to its (non-null) children
  void RenderTable(const EventType *query, int32_t id);

  /// Makes a comma-separated string from the elements of a set of identifiers
  /// If the identifiers represent phones, their symbolic representations are
  /// used
  std::string MakeEdgeLabel(const EventKeyType &key,
                            const ConstIntegerSet<EventValueType> &intset);

  /// Writes the GraphViz representation of a non-leaf node to the out stream
  /// A question about a phone from the context window or about pdf-class
  /// is used as a label.
  void RenderNonLeaf(int32_t id, const EventKeyType &key, bool in_query);

  fst::SymbolTable &phone_syms_;  // phone symbols to be used as edge labels
  std::istream &is_;              // the stream from which the tree is read
  std::ostream &out_;  // the GraphViz representation is written to this stream
  bool binary_;        // is the input stream binary?
  int32_t N_, P_;      // context-width and central position
  bool use_tooltips_;  // use tooltips(useful in e.g. SVG) instead of labels
  int32_t next_id_;    // the first unused GraphViz node ID
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TREE_RENDERER_H_
