# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reimplementation of Kaldi's HMM-GMM with Eigen. Architecture contract:

- **User interface = Python API** (`kaldi_hmm_gmm` package); **implementation = C++** in `kaldi-hmm-gmm/csrc/`, ported from `../kaldi` (a local kaldi checkout that is the reference for everything).
- FST operations come from the sibling project `kaldifst`; ark/scp I/O from `kaldi_native_io` (pip). Neither is a kaldi binary — this repo must not shell out to kaldi binaries at runtime. `../kaldi` is used only as reference source, test-data oracle, and (for egs) feature pre-computation.

## Build & test

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2                      # CI uses -j2; keep it modest
ctest --output-on-failure     # runs both gtest and python tests (python tests are <name>_py)
```

- Single test: `cd build && ./check.sh <regexp>` (builds + `ctest -R <regexp>`), or run a binary directly: `build/bin/diag-gmm-test`. Python tests live in `kaldi-hmm-gmm/python/tests/test_*.py` and are registered with ctest via `kgh_add_py_test` (note: this name is a typo in `python/tests/CMakeLists.txt` — the repo abbreviation is `khg`; rename to `khg_add_py_test` is planned in Stage 0).
- New gtest: add `xxx-test.cc` to `test_srcs` in `kaldi-hmm-gmm/csrc/CMakeLists.txt` (uses `khg_add_test()`).
- Python package/wheel: `python setup.py install` (cmake-based `cmake_extension`; also builds `_kaldi_hmm_gmm`). Note `setup.py` temporarily appends `__version__` to `kaldi_hmm_gmm/__init__.py` — don't edit that file concurrently.
- Lint (C++): `./scripts/check_style_cpplint.sh` (`1` = uncommitted changes, `2` = whole project).
- Building the reference kaldi: `cd ../kaldi/src && make` or `make -j2` — **never more than 2 jobs**.

## Architecture map

- `kaldi-hmm-gmm/csrc/` — C++ core, mirrors `../kaldi/src/{hmm,gmm,decoder,tree,itf,...}` (e.g. `transition-model`, `hmm-topology`, `hmm-utils`, `diag-gmm`, `mle-diag-gmm`, `training-graph-compiler`, `decoder-wrappers`, `lattice-faster-decoder`). Keep file/function names and semantics aligned with kaldi.
- `kaldi-hmm-gmm/python/csrc/` — pybind11 bindings, one `Pybind*.cc`-style file per C++ module, aggregated in `kaldi-hmm-gmm.cc`. The extension module is built **into the package**: `kaldi_hmm_gmm/lib/_kaldi_hmm_gmm.*` (self-contained: `kaldi-hmm-gmm-core` is always built STATIC and linked in). `kaldi_hmm_gmm/__init__.py` re-exports from `kaldi_hmm_gmm.lib._kaldi_hmm_gmm`.
- `scripts/` — Python equivalents of kaldi binaries (`gmm_init_mono.py`, `gmm_align_compiled.py`, `gmm_acc_stats_ali.py`, `gmm_est.py`, `gmm_boost_silence.py`, `prepare_lang.py`). **Being dissolved** (plan.md §4.0): library code moves into the `kaldi_hmm_gmm` package, tests into `python/tests/`, bash helpers into `egs/utils/`; its current content is NOT authoritative — per user instruction it may be deleted or rewritten freely (e.g. drop lhotse from signatures).
- `egs/yesno/` — recipe: `run.sh` stages drive `train.py` (mono GMM training loop: compile graphs → equal-align / re-align → acc-stats → gmm-est with mixup) and `decode.py` (L·G→CLG→HCLG + lattice-faster decode). Model persists as kaldi-format `final.mdl` (`AmDiagGmm` + `TransitionModel` + `ContextDependency` via the kaldi-compatible Read/Write).
- Training graph pipeline (both kaldi and here): graph compiles WITH transition probs baked, then `AddTransitionProbs` is applied again before alignment — do not "fix" this double application; it matches kaldi.

## Standing working rules (user's instructions)

1. **Fill gaps from kaldi**: if C++ is missing functionality, port it from `../kaldi`'s corresponding source. If the Python API is missing something, add pybind bindings against the C++. Never work around a C++ gap in Python.
2. **Verify every step early — never batch tests to the end.** After each C++ module **and after each pipeline stage**: (a) line-compare the C++ with `../kaldi` source; (b) add gtest in this repo; (c) **numerically compare this repo's output vs kaldi's on identical input** (kaldi binaries are the oracle here; exchange data via `kaldi_native_io`; parity artifacts under `exp/parity/`). **For stages with randomness, use stage-substitution**: inject kaldi-generated artifacts as that stage's output (features/text/alignments are directly readable; models need `DiagGmm`/`AmDiagGmm` Read/Write ported+bound), verify downstream deterministic stages on identical input, then swap back to our own outputs one stage at a time (ratchet). Checklists in `plan.md` §4.7–4.9 (§4.9 is the per-step gate: gtest green + kaldi parity + discrepancies resolved + no batching — all must pass before the next step). Fix or report discrepancies before proceeding to the next step.
3. **Parity comparison policy**: ignore floating-point precision differences and random-number effects (e.g. `align-equal-compiled` seeds, dither). Report any remaining real behavioral difference to the user and let them decide — do not silently align configs.
4. **Don't write code before the user approves the plan** (plans live in `plan.md`; the user reviews and explicitly approves).
5. Feature computation is **not** this repo's job (for now): use kaldi-precomputed features read via `kaldi_native_io`.
6. **Shared Python functions go in the package** `kaldi-hmm-gmm/python/kaldi_hmm_gmm/` — never in `scripts/`. The user will publish `kaldi-hmm-gmm` as a proper Python package. Recipe-specific glue stays in `egs/<recipe>/`; dev/test tooling in `tests/` or `tools/`.
7. **Existing C++ and Python implementations are UNTRUSTED — they may contain bugs** (user's warning). Even code nominally copied from `../kaldi` is suspect: proactively source-diff each module against `../kaldi`, run it through the §4.9 gate, and on any parity mismatch suspect this repo's implementation first (old code included); fix and report to the user. kaldi is the only oracle.
8. **No lhotse dependency, anywhere** (hard constraint): package API, egs scripts, and tests must not import lhotse. Existing `scripts/` content may be modified or deleted freely — anything lhotse-based (e.g. `gmm_init_mono.py` taking CutSets, `compute_fbank_*`) gets rewritten around `Dict[utt, np.ndarray]` / kaldi-style `text`+`feats.scp`, or removed.
9. **`egs/yesno` may be deleted and rewritten from scratch** to mirror `../kaldi/egs/yesno/s5`'s structure (run.sh stages matching kaldi's one-to-one), each step calling the `kaldi_hmm_gmm` package. The existing recipe is not authoritative.

## Local environment (this machine)

- Python env `~/py312` (Python 3.12): has `kaldifst`, `kaldi_native_io`, `kaldi_native_fbank`, lhotse (editable). **No torch** (the recipe/tools/tests are numpy-only). `kaldi_hmm_gmm` itself must be built for the env in use (check the tag of `kaldi-hmm-gmm/python/kaldi_hmm_gmm/lib/_kaldi_hmm_gmm.cpython-*.so`).
- (no `path.sh` needed any more: the scripts insert `kaldi-hmm-gmm/python` and `tools/` into `sys.path` themselves; everything else is pip-installed.)
- Sibling repos: `../kaldi` (reference), `../kaldifst`, `../kaldi_native_io`, `../kaldi-native-fbank` (all local checkouts).
- **lhotse is NOT used** (decision): data layer is kaldi-style `text`/`utt2spk`/`feats.scp` read via `kaldi_native_io` + plain text. lhotse is installed (editable → `../lhotse`) but the pipeline must not depend on it.

## Current work (pick up here)

Task: forced alignment for yesno **without kaldi binaries** — kaldi's `train_mono.sh` + `ali-to-phones --per-frame=true` equivalent in this repo (see `todo.md` for the kaldi reference output to match).

- `plan.md` is the full design (gap analysis vs kaldi, staged implementation Stage 0 restructure → A/A'/B/C/D → **E parity, which is mandatory**). §7 is now a **decisions record** (all resolved — do not re-ask the user).
- Decisions (final): **mirror `../kaldi/egs/yesno/s5` exactly** — kaldi egs/yesno has it ⇒ we implement it; file names, save locations, dir layout follow kaldi (`exp/mono0a/`, `graph_tgpr/`, `decode_test_yesno/`, `log/`, per-iter `N.mdl`/`ali.1.gz`/`*.occs`, …); the ONLY exception is features (precomputed via `egs/yesno/local/compute_feats.sh`, stable contract `data/feats/{train,test}_feats.{ark,scp}`, already run: train 31 / test 29, dim 39). All training configs align to kaldi (31 utts no perturb, 40 iters, totgauss 400, power 0.25, beam 6→10, realign schedule, init stats n=10). No lhotse. Output per-frame phones in kaldi `ark,t:` format via new C++ `SplitToPhones` port + package API. Random stages are verified by kaldi-artifact injection (plan.md §4.8). Where the old repo example disagrees with kaldi, kaldi wins; the old `scripts/` and `egs/yesno` content is free to delete/rewrite (don't ask how to handle it); the audio-mnist example is ignored (user maintains examples themselves later). Note: kaldi yesno split is train 31 / test 29 (total 60).
- Known kaldi-vs-repo difference inventory is in `plan.md` §3.7.2; float/random diffs are excluded by policy, everything else gets aligned to kaldi.
