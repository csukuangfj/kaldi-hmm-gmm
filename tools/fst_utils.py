# Copyright (c)  2026  Xiaomi Corporation
#
# Small fst helpers for recipe logging (mirrors kaldi's fstisstochastic).

import math


def fstis_stochastic(fst):
    """kaldi's fstisstochastic: returns (min_sum, max_sum) over states of
    -log(sum of outgoing-arc + final probabilities) (0 => stochastic).

    The fst is read from its OpenFst text form (fst.to_str())."""
    arcs = {}
    finals = {}
    for line in fst.to_str().splitlines():
        parts = line.split()
        if len(parts) == 1:
            finals[int(parts[0])] = 0.0
        elif len(parts) == 2:
            finals[int(parts[0])] = float(parts[1])
        elif len(parts) in (4, 5):
            src = int(parts[0])
            w = float(parts[4]) if len(parts) == 5 else 0.0
            arcs.setdefault(src, []).append(w)
    states = set(arcs) | set(finals)
    sums = []
    for s in states:
        tot = math.exp(-finals.get(s, float("inf")))
        for w in arcs.get(s, []):
            tot += math.exp(-w)
        sums.append(-math.log(tot) if tot > 0 else float("inf"))
    return min(sums), max(sums)
