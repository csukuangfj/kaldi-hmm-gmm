#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation
#
# Smoke test for the built wheels. Run by build-wheels-win-arm64.yaml and
# build-wheels-android-termux.yaml after installing a freshly built wheel.
#
# It builds a small HMM system and queries it so that a wheel which cannot
# load its shared libraries (bad RPATH, missing DLL, wrong architecture)
# fails here rather than on a user's machine.
#
# numpy is optional: the topology/tree/transition-model path needs only
# plain Python lists, so this also runs inside the Termux images where
# numpy may not be packaged.

import platform
import sys

import kaldi_hmm_gmm as khg

TOPOLOGY = """
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 </ForPhones>
 <State> 0 <PdfClass> 0
 <Transition> 0 0.5
 <Transition> 1 0.5
 </State>
 <State> 1 <PdfClass> 1
 <Transition> 1 0.5
 <Transition> 2 0.5
 </State>
 <State> 2 <PdfClass> 2
 <Transition> 2 0.5
 <Transition> 3 0.5
 </State>
 <State> 3
 </State>
 </TopologyEntry>
 </Topology>
"""


def main() -> int:
    print("kaldi_hmm_gmm:", khg.__file__)
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("machine:", platform.machine())

    topo = khg.HmmTopology()
    topo.read(TOPOLOGY)
    phones = topo.phones
    print("phones:", list(phones))

    phone2num_pdf_classes = [0] * (phones[-1] + 1)
    for p in phones:
        phone2num_pdf_classes[p] = topo.num_pdf_classes(p)
    ctx_dep = khg.monophone_context_dependency(
        phones=phones, phone2num_pdf_classes=phone2num_pdf_classes
    )
    tm = khg.TransitionModel(ctx_dep=ctx_dep, hmm_topo=topo)

    num_pdfs = ctx_dep.num_pdfs
    num_tids = tm.num_transition_ids()
    print("num_pdfs:", num_pdfs, "num_transition_ids:", num_tids)
    assert num_pdfs > 0, "expected at least one pdf"
    assert num_tids > 0, "expected at least one transition-id"
    assert tm.transition_id_to_phone(1) in phones, "bad phone mapping"

    try:
        import numpy as np
    except ImportError:
        print("numpy is not installed; skipping the GMM likelihood check")
    else:
        dim = 3
        gmm = khg.DiagGmm(nmix=1, dim=dim)
        gmm.set_weights(np.ones(1, dtype=np.float32))
        gmm.set_invvars_and_means(
            np.ones((1, dim), dtype=np.float32),
            np.zeros((1, dim), dtype=np.float32),
        )
        gmm.compute_gconsts()
        like = gmm.log_likelihood(np.zeros(dim, dtype=np.float32))
        print("log_likelihood:", like)
        assert np.isfinite(like), "likelihood is not finite"

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
