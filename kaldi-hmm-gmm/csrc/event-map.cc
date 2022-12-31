// kaldi-hmm-gmm/csrc/event-map.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/event-map.h"

#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi_native_io/csrc/io-funcs.h"

namespace khg {

void WriteEventType(std::ostream &os, bool binary, const EventType &evec) {
  kaldiio::WriteToken(os, binary, "EV");
  uint32_t size = evec.size();
  kaldiio::WriteBasicType(os, binary, size);
  for (size_t i = 0; i < size; ++i) {
    kaldiio::WriteBasicType(os, binary, evec[i].first);
    kaldiio::WriteBasicType(os, binary, evec[i].second);
  }

  if (!binary) {
    os << '\n';
  }
}

void ReadEventType(std::istream &is, bool binary, EventType *evec) {
  KHG_ASSERT(evec != NULL);
  kaldiio::ExpectToken(is, binary, "EV");
  uint32_t size;

  kaldiio::ReadBasicType(is, binary, &size);
  evec->resize(size);
  for (size_t i = 0; i < size; i++) {
    kaldiio::ReadBasicType(is, binary, &((*evec)[i].first));
    kaldiio::ReadBasicType(is, binary, &((*evec)[i].second));
  }
}

std::string EventTypeToString(const EventType &evec) {
  std::stringstream ss;
  EventType::const_iterator iter = evec.begin(), end = evec.end();
  std::string sep = "";
  for (; iter != end; ++iter) {
    ss << sep << iter->first << ":" << iter->second;
    sep = " ";
  }
  return ss.str();
}

}  // namespace khg
