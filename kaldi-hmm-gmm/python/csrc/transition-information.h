// kaldi-hmm-gmm/python/csrc/transition-information.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Registration of kaldi-decoder's TransitionInformation in THIS module
// (module_local): kaldi-hmm-gmm is self-contained at the Python level and
// does not depend on the kaldi-decoder Python package.

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_

#include <cstring>
#include <string>
#include <typeinfo>

#include "kaldi-decoder/csrc/transition-information.h"
#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"
#include "pybind11/conduit/pybind11_platform_abi_id.h"

namespace khg {

void PybindTransitionInformation(py::module *m);

// Cross-module bridge for kaldi_decoder::TransitionInformation.
//
// Ported from ../kaldi-decoder/python/csrc/transition-information.h (keep
// in sync).  pybind11's class_ machinery installs a default
// `_pybind11_conduit_v1_` provider that locates the requested C++ type
// through its std::type_info identity -- an identity that is per-module
// when the wheels hide symbols, so a kaldi_hmm_gmm.TransitionModel cannot
// reach kaldi-decoder bindings (and vice versa).  The conduit protocol
// (pybind11/conduit/pybind11_conduit_v1.h) explicitly lets bindings "roll
// their own" provider; this one matches the requested type by its mangled
// NAME (stable across modules) and answers only for
// TransitionInformation.  Anything else falls through to pybind11's
// default provider.
//
// Contract: every module in the process must be built against the same
// transition-information.h (the object stays owned by the module that
// created it; we only borrow its address).
inline py::object TransitionInformationConduitMethod(
    py::handle self, const py::bytes &pybind11_platform_abi_id,
    const py::capsule &cpp_type_info_capsule, const py::bytes &pointer_kind) {
#ifdef PYBIND11_HAS_STRING_VIEW
  using cpp_str = std::string_view;
#else
  using cpp_str = std::string;
#endif
  if (cpp_str(pybind11_platform_abi_id) == PYBIND11_PLATFORM_ABI_ID &&
      std::strcmp(cpp_type_info_capsule.name(),
                  typeid(std::type_info).name()) == 0 &&
      cpp_str(pointer_kind) == "raw_pointer_ephemeral") {
    const auto *requested =
        cpp_type_info_capsule.get_pointer<const std::type_info>();
    if (requested != nullptr &&
        std::strcmp(requested->name(),
                    typeid(kaldi_decoder::TransitionInformation).name()) == 0) {
      auto *ti = py::cast<const kaldi_decoder::TransitionInformation *>(self);
      return py::capsule(const_cast<void *>(static_cast<const void *>(ti)),
                         requested->name());
    }
  }
  return pybind11::detail::cpp_conduit_method(
      self, pybind11_platform_abi_id, cpp_type_info_capsule, pointer_kind);
}

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_
