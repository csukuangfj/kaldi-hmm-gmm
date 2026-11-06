# Copyright (c)  2026  Xiaomi Corporation
#
# Python equivalent of kaldi's bin/ali-to-phones.
# See ../kaldi/src/bin/ali-to-phones.cc
#
# Usage:
#   from ali_to_phones import ali_to_phones, int2sym

from typing import Dict, List, Tuple, Union

import kaldi_hmm_gmm as khg


def split_to_phones(
    transition_model: "khg.TransitionModel", ali: List[int]
) -> List[List[int]]:
    """Split a transition-id alignment into per-phone segments.

    This wraps khg.split_to_phones (C++, ported from ../kaldi's
    SplitToPhones).  Adjacent instances of the same phone are NOT merged.

    Returns:
      A list of segments; each segment is the list of transition-ids of one
      phone instance.
    """
    ok, split_alignment = khg.split_to_phones(transition_model, ali)
    if not ok:
        # kaldi still splits incomplete alignments; ali-to-phones just writes
        # whatever SplitToPhones produced.  Keep the same behavior.
        pass
    return split_alignment


def ali_to_phones(
    transition_model: "khg.TransitionModel",
    ali: List[int],
    per_frame: bool = True,
    write_lengths: bool = False,
) -> Union[List[int], List[Tuple[int, int]]]:
    """Convert a transition-id alignment to a phone sequence.

    Mirrors kaldi's ali-to-phones options:

    Args:
      transition_model: the TransitionModel that produced the alignment.
      ali: list of transition-ids (one per frame).
      per_frame: if True, write one phone-id per frame; else one phone-id per
        phone segment.
      write_lengths: if True, return [(phone_id, num_frames), ...] instead.
        Incompatible with per_frame.

    Returns:
      per_frame=True  -> List[int], length == num frames
      per_frame=False -> List[int], one phone-id per phone instance
      write_lengths   -> List[Tuple[int, int]]
    """
    assert len(ali) == 0 or isinstance(ali[0], int)
    assert not (per_frame and write_lengths), "incompatible options"

    split_alignment = split_to_phones(transition_model, ali)

    if write_lengths:
        return [
            (transition_model.transition_id_to_phone(seg[0]), len(seg))
            for seg in split_alignment
        ]

    phones: List[int] = []
    for seg in split_alignment:
        phone = transition_model.transition_id_to_phone(seg[0])
        if per_frame:
            phones.extend([phone] * len(seg))
        else:
            phones.append(phone)
    return phones


def int2sym(seq: List[int], id2sym: Dict[int, str]) -> List[str]:
    """Map integer ids to symbols (kaldi's utils/int2sym.pl equivalent)."""
    return [id2sym[i] for i in seq]


def read_sym_table(path: str) -> Dict[int, str]:
    """Read a kaldi symbol table file (lines of 'sym id'); returns id -> sym."""
    id2sym: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            sym, idx = parts
            id2sym[int(idx)] = sym
    return id2sym
