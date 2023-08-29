// kaldi-hmm-gmm/csrc/hmm-topology.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// This file is copied and modified from
// kaldi/src/hmm/hmm-topology.cc
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>

#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/text-utils.h"

namespace khg {

void HmmTopology::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "<Topology>");

  if (!binary) {  // Text-mode read, different "human-readable" format.
    phones_.clear();
    phone2idx_.clear();
    entries_.clear();

    std::string token;
    while (!(is >> token).fail()) {
      if (token == "</Topology>") {
        // finished parsing.
        break;
      } else if (token != "<TopologyEntry>") {
        KHG_ERR << "Reading HmmTopology object, expected </Topology> or "
                   "<TopologyEntry>, got "
                << token;
      } else {
        kaldiio::ExpectToken(is, binary, "<ForPhones>");
        std::vector<int32_t> phones;
        std::string s;
        while (true) {
          is >> s;
          if (is.fail())
            KHG_ERR << "Reading HmmTopology object, unexpected end of file "
                       "while expecting phones.";
          if (s == "</ForPhones>") {
            break;
          } else {
            int32_t phone;
            if (!kaldiio::ConvertStringToInteger(s, &phone))
              KHG_ERR << "Reading HmmTopology object, expected "
                      << "integer, got instead " << s;
            phones.push_back(phone);
          }
        }  // while (true)

        std::vector<HmmState> this_entry;
        std::string token;
        kaldiio::ReadToken(is, binary, &token);
        while (token != "</TopologyEntry>") {
          if (token != "<State>")
            KHG_ERR << "Expected </TopologyEntry> or <State>, got instead "
                    << token;
          int32_t state;
          kaldiio::ReadBasicType(is, binary, &state);
          if (state != static_cast<int32_t>(this_entry.size()))
            KHG_ERR << "States are expected to be in order from zero, expected "
                    << this_entry.size() << ", got " << state;
          kaldiio::ReadToken(is, binary, &token);
          int32_t forward_pdf_class = kNoPdf;  // -1 by default, means no pdf.
          if (token == "<PdfClass>") {
            kaldiio::ReadBasicType(is, binary, &forward_pdf_class);
            this_entry.push_back(HmmState(forward_pdf_class));
            kaldiio::ReadToken(is, binary, &token);
            if (token == "<SelfLoopPdfClass>") {
              KHG_ERR << "pdf classes should be defined using <PdfClass> "
                      << "or <ForwardPdfClass>/<SelfLoopPdfClass> pair";
            }
          } else if (token == "<ForwardPdfClass>") {
            int32_t self_loop_pdf_class = kNoPdf;
            kaldiio::ReadBasicType(is, binary, &forward_pdf_class);
            kaldiio::ReadToken(is, binary, &token);
            if (token != "<SelfLoopPdfClass>") {
              KHG_ERR << "Expected <SelfLoopPdfClass>, got instead " << token;
            }

            kaldiio::ReadBasicType(is, binary, &self_loop_pdf_class);
            this_entry.push_back(
                HmmState(forward_pdf_class, self_loop_pdf_class));
            kaldiio::ReadToken(is, binary, &token);
          } else {
            this_entry.push_back(HmmState(forward_pdf_class));
          }

          while (token == "<Transition>") {
            int32_t dst_state;
            float trans_prob;
            kaldiio::ReadBasicType(is, binary, &dst_state);
            kaldiio::ReadBasicType(is, binary, &trans_prob);
            this_entry.back().transitions.push_back(
                std::make_pair(dst_state, trans_prob));
            kaldiio::ReadToken(is, binary, &token);
          }

          if (token == "<Final>") {
            // TODO(fangjun): remove this clause after a while.
            KHG_ERR
                << "You are trying to read old-format topology with new Kaldi.";
          }

          if (token != "</State>") {
            KHG_ERR << "Expected </State>, got instead " << token;
          }

          kaldiio::ReadToken(is, binary, &token);
        }  // while (token != "</TopologyEntry>")

        int32_t my_index = entries_.size();
        entries_.push_back(this_entry);

        for (size_t i = 0; i < phones.size(); i++) {
          int32_t phone = phones[i];
          if (static_cast<int32_t>(phone2idx_.size()) <= phone) {
            phone2idx_.resize(phone + 1, -1);  // -1 is invalid index.
          }

          KHG_ASSERT(phone > 0);

          if (phone2idx_[phone] != -1) {
            KHG_ERR << "Phone with index " << i
                    << " appears in multiple topology entries.";
          }

          phone2idx_[phone] = my_index;
          phones_.push_back(phone);
        }
      }
    }  // while (!(is >> token).fail())
    std::sort(phones_.begin(), phones_.end());
    KHG_ASSERT(IsSortedAndUniq(phones_));
  } else {  // binary I/O, just read member objects directly from disk.
    ReadIntegerVector(is, binary, &phones_);
    ReadIntegerVector(is, binary, &phone2idx_);
    int32_t sz;
    kaldiio::ReadBasicType(is, binary, &sz);
    bool is_hmm = true;
    if (sz == -1) {
      is_hmm = false;
      kaldiio::ReadBasicType(is, binary, &sz);
    }

    entries_.resize(sz);
    for (int32_t i = 0; i < sz; ++i) {
      int32_t thist_sz;
      kaldiio::ReadBasicType(is, binary, &thist_sz);
      entries_[i].resize(thist_sz);
      for (int32_t j = 0; j < thist_sz; ++j) {
        kaldiio::ReadBasicType(is, binary, &(entries_[i][j].forward_pdf_class));
        if (is_hmm) {
          entries_[i][j].self_loop_pdf_class = entries_[i][j].forward_pdf_class;
        } else {
          kaldiio::ReadBasicType(is, binary,
                                 &(entries_[i][j].self_loop_pdf_class));
        }

        int32_t thiss_sz;
        kaldiio::ReadBasicType(is, binary, &thiss_sz);
        entries_[i][j].transitions.resize(thiss_sz);
        for (int32_t k = 0; k < thiss_sz; ++k) {
          kaldiio::ReadBasicType(is, binary,
                                 &(entries_[i][j].transitions[k].first));
          kaldiio::ReadBasicType(is, binary,
                                 &(entries_[i][j].transitions[k].second));
        }
      }  // for (int32_t j = 0; j < thist_sz; ++j)
    }    // for (int32_t i = 0; i < sz; ++i)

    kaldiio::ExpectToken(is, binary, "</Topology>");
  }
  Check();  // Will throw if not ok.
}

void HmmTopology::Write(std::ostream &os, bool binary) const {
  bool is_hmm = IsHmm();
  kaldiio::WriteToken(os, binary, "<Topology>");
  if (!binary) {  // Text-mode write.
    os << "\n";
    for (int32_t i = 0; i < static_cast<int32_t>(entries_.size()); ++i) {
      kaldiio::WriteToken(os, binary, "<TopologyEntry>");
      os << "\n";
      kaldiio::WriteToken(os, binary, "<ForPhones>");
      os << "\n";
      for (size_t j = 0; j < phone2idx_.size(); ++j) {
        if (phone2idx_[j] == i) {
          os << j << " ";
        }
      }

      os << "\n";
      kaldiio::WriteToken(os, binary, "</ForPhones>");
      os << "\n";

      for (size_t j = 0; j < entries_[i].size(); ++j) {
        kaldiio::WriteToken(os, binary, "<State>");
        kaldiio::WriteBasicType(os, binary, static_cast<int32_t>(j));

        if (entries_[i][j].forward_pdf_class != kNoPdf) {
          if (is_hmm) {
            kaldiio::WriteToken(os, binary, "<PdfClass>");
            kaldiio::WriteBasicType(os, binary,
                                    entries_[i][j].forward_pdf_class);
          } else {
            kaldiio::WriteToken(os, binary, "<ForwardPdfClass>");
            kaldiio::WriteBasicType(os, binary,
                                    entries_[i][j].forward_pdf_class);
            KHG_ASSERT(entries_[i][j].self_loop_pdf_class != kNoPdf);
            kaldiio::WriteToken(os, binary, "<SelfLoopPdfClass>");
            kaldiio::WriteBasicType(os, binary,
                                    entries_[i][j].self_loop_pdf_class);
          }
        }

        for (size_t k = 0; k < entries_[i][j].transitions.size(); ++k) {
          kaldiio::WriteToken(os, binary, "<Transition>");
          kaldiio::WriteBasicType(os, binary,
                                  entries_[i][j].transitions[k].first);
          kaldiio::WriteBasicType(os, binary,
                                  entries_[i][j].transitions[k].second);
        }

        kaldiio::WriteToken(os, binary, "</State>");
        os << "\n";
      }

      kaldiio::WriteToken(os, binary, "</TopologyEntry>");
      os << "\n";
    }
  } else {
    // for binary
    WriteIntegerVector(os, binary, phones_);
    WriteIntegerVector(os, binary, phone2idx_);
    // -1 is put here as a signal that the object has the new,
    // extended format with SelfLoopPdfClass
    if (!is_hmm) {
      kaldiio::WriteBasicType(os, binary, static_cast<int32_t>(-1));
    }

    kaldiio::WriteBasicType(os, binary, static_cast<int32_t>(entries_.size()));

    for (size_t i = 0; i < entries_.size(); ++i) {
      kaldiio::WriteBasicType(os, binary,
                              static_cast<int32_t>(entries_[i].size()));

      for (size_t j = 0; j < entries_[i].size(); ++j) {
        kaldiio::WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);

        if (!is_hmm) {
          kaldiio::WriteBasicType(os, binary,
                                  entries_[i][j].self_loop_pdf_class);
        }

        kaldiio::WriteBasicType(
            os, binary,
            static_cast<int32_t>(entries_[i][j].transitions.size()));

        for (size_t k = 0; k < entries_[i][j].transitions.size(); ++k) {
          kaldiio::WriteBasicType(os, binary,
                                  entries_[i][j].transitions[k].first);
          kaldiio::WriteBasicType(os, binary,
                                  entries_[i][j].transitions[k].second);
        }
      }
    }
  }
  kaldiio::WriteToken(os, binary, "</Topology>");
  if (!binary) {
    os << "\n";
  }
}

bool HmmTopology::IsHmm() const {
  const std::vector<int32_t> &phones = GetPhones();
  KHG_ASSERT(!phones.empty());
  for (size_t i = 0; i < phones.size(); ++i) {
    int32_t phone = phones[i];
    const TopologyEntry &entry = TopologyForPhone(phone);
    for (int32_t j = 0; j < static_cast<int32_t>(entry.size());
         ++j) {  // for each state...
      int32_t forward_pdf_class = entry[j].forward_pdf_class,
              self_loop_pdf_class = entry[j].self_loop_pdf_class;

      if (forward_pdf_class != self_loop_pdf_class) {
        return false;
      }
    }
  }
  return true;
}

const HmmTopology::TopologyEntry &HmmTopology::TopologyForPhone(
    int32_t phone) const {  // Will throw if phone not covered.
  if (static_cast<size_t>(phone) >= phone2idx_.size() ||
      phone2idx_[phone] == -1) {
    KHG_ERR << "TopologyForPhone(), phone " << phone << " not covered.";
  }
  return entries_[phone2idx_[phone]];
}

void HmmTopology::Check() {
  if (entries_.empty() || phones_.empty() || phone2idx_.empty()) {
    KHG_ERR << "HmmTopology::Check(), empty object.";
  }

  std::vector<bool> is_seen(entries_.size(), false);
  for (size_t i = 0; i < phones_.size(); ++i) {
    int32_t phone = phones_[i];
    if (static_cast<size_t>(phone) >= phone2idx_.size() ||
        static_cast<size_t>(phone2idx_[phone]) >= entries_.size()) {
      KHG_ERR << "HmmTopology::Check(), phone has no valid index.";
    }

    is_seen[phone2idx_[phone]] = true;
  }
  for (size_t i = 0; i < entries_.size(); i++) {
    if (!is_seen[i]) {
      KHG_ERR << "HmmTopoloy::Check(), entry with no corresponding phones.";
    }

    int32_t num_states = static_cast<int32_t>(entries_[i].size());
    if (num_states <= 1) {
      KHG_ERR << "HmmTopology::Check(), cannot only have one state (i.e., must "
                 "have at least one emitting state).";
    }

    if (!entries_[i][num_states - 1].transitions.empty()) {
      KHG_ERR << "HmmTopology::Check(), last state must have no transitions.";
    }

    // not sure how necessary this next stipulation is.
    if (entries_[i][num_states - 1].forward_pdf_class != kNoPdf) {
      KHG_ERR << "HmmTopology::Check(), last state must not be emitting.";
    }

    std::vector<bool> has_trans_in(num_states, false);
    std::vector<int32_t> seen_pdf_classes;

    for (int32_t j = 0; j < num_states; ++j) {  // j is the state-id.
      float tot_prob = 0.0;
      if (entries_[i][j].forward_pdf_class != kNoPdf) {
        seen_pdf_classes.push_back(entries_[i][j].forward_pdf_class);
        seen_pdf_classes.push_back(entries_[i][j].self_loop_pdf_class);
      }
      std::set<int32_t> seen_transition;
      for (int32_t k = 0;
           static_cast<size_t>(k) < entries_[i][j].transitions.size(); ++k) {
        tot_prob += entries_[i][j].transitions[k].second;

        if (entries_[i][j].transitions[k].second <= 0.0) {
          KHG_ERR << "HmmTopology::Check(), negative or zero transition prob.";
        }

        int32_t dst_state = entries_[i][j].transitions[k].first;

        // The commented code in the next few lines disallows a completely
        // skippable phone, as this would cause to stop working some mechanisms
        // that are being built, which enable the creation of phone-level
        // lattices and rescoring these with a different lexicon and LM.
        if (dst_state == num_states - 1  // && j != 0
            && entries_[i][j].forward_pdf_class == kNoPdf) {
          KHG_ERR << "We do not allow any state to be "
                     "nonemitting and have a transition to the final-state "
                     "(this would "
                     "stop the SplitToPhones function from identifying the "
                     "last state "
                     "of a phone.";
        }

        if (dst_state < 0 || dst_state >= num_states) {
          KHG_ERR << "HmmTopology::Check(), invalid dest state " << (dst_state);
        }

        if (seen_transition.count(dst_state) != 0) {
          KHG_ERR << "HmmTopology::Check(), duplicate transition found.";
        }

        if (dst_state == k) {  // self_loop...
          KHG_ASSERT(entries_[i][j].self_loop_pdf_class != kNoPdf &&
                     "Nonemitting states cannot have self-loops.");
        }

        seen_transition.insert(dst_state);
        has_trans_in[dst_state] = true;
      }

      if (j + 1 < num_states) {
        KHG_ASSERT(tot_prob > 0.0 &&
                   "Non-final state must have transitions out."
                   "(with nonzero probability)");
        if (std::fabs(tot_prob - 1.0) > 0.01) {
          KHG_WARN << "Total probability for state " << j
                   << " in topology entry is " << tot_prob;
        }
      } else {
        KHG_ASSERT(tot_prob == 0.0);
      }
    }

    // make sure all but start state have input transitions.
    for (int32_t j = 1; j < num_states; ++j) {
      if (!has_trans_in[j]) {
        KHG_ERR << "HmmTopology::Check, state " << (j)
                << " has no input transitions.";
      }
    }

    SortAndUniq(&seen_pdf_classes);
    if (seen_pdf_classes.front() != 0 ||
        seen_pdf_classes.back() !=
            static_cast<int32_t>(seen_pdf_classes.size()) - 1) {
      KHG_ERR << "HmmTopology::Check(), pdf_classes are expected to be "
                 "contiguous and start from zero.";
    }
  }
}

int32_t HmmTopology::NumPdfClasses(int32_t phone) const {
  // will throw if phone not covered.
  const TopologyEntry &entry = TopologyForPhone(phone);
  int32_t max_pdf_class = 0;
  for (size_t i = 0; i < entry.size(); ++i) {
    max_pdf_class = std::max(max_pdf_class, entry[i].forward_pdf_class);
    max_pdf_class = std::max(max_pdf_class, entry[i].self_loop_pdf_class);
  }
  return max_pdf_class + 1;
}

void HmmTopology::GetPhoneToNumPdfClasses(
    std::vector<int32_t> *phone2num_pdf_classes) const {
  KHG_ASSERT(!phones_.empty());

  phone2num_pdf_classes->clear();
  phone2num_pdf_classes->resize(phones_.back() + 1, -1);

  for (size_t i = 0; i < phones_.size(); ++i)
    (*phone2num_pdf_classes)[phones_[i]] = NumPdfClasses(phones_[i]);
}

int32_t HmmTopology::MinLength(int32_t phone) const {
  const TopologyEntry &entry = TopologyForPhone(phone);

  // min_length[state] gives the minimum length for sequences up to and
  // including that state.
  std::vector<int32_t> min_length(entry.size(),
                                  std::numeric_limits<int32_t>::max());
  KHG_ASSERT(!entry.empty());

  min_length[0] = (entry[0].forward_pdf_class == -1 ? 0 : 1);

  int32_t num_states = min_length.size();
  bool changed = true;
  while (changed) {
    changed = false;
    for (int32_t s = 0; s < num_states; ++s) {
      const HmmState &this_state = entry[s];

      std::vector<std::pair<int32_t, float>>::const_iterator
          iter = this_state.transitions.begin(),
          end = this_state.transitions.end();

      for (; iter != end; ++iter) {
        int32_t next_state = iter->first;
        KHG_ASSERT(next_state < num_states);

        int32_t next_state_min_length =
            min_length[s] + (entry[next_state].forward_pdf_class == -1 ? 0 : 1);
        if (next_state_min_length < min_length[next_state]) {
          min_length[next_state] = next_state_min_length;
          if (next_state < s) {
            changed = true;
          }
          // the test of 'next_state < s' is an optimization for speed.
        }
      }
    }
  }
  KHG_ASSERT(min_length.back() != std::numeric_limits<int32_t>::max());
  // the last state is the final-state.
  return min_length.back();
}

}  // namespace khg
