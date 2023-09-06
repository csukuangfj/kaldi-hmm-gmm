# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

import kaldifst


def build_standard_ctc_topo(max_token_id: int) -> kaldifst.StdVectorFst:
    """
    Args:
      Maximum valid token ID. We assume token IDs are contiguous
      and starts from 0. In otherwords, the vocabulary size is
      ``max_token_id + 1``.
    """
    num_states = max_token_id + 1

    fst = kaldifst.StdVectorFst()
    for i in range(num_states):
        s = fst.add_state()
        fst.set_final(state=s, weight=0)
    fst.start = 0

    # fully connected
    for i in range(num_states):
        for k in range(num_states):
            fst.add_arc(
                state=i,
                arc=kaldifst.StdArc(
                    ilabel=k,
                    olabel=k if i != k else 0,  # if i==k, it is a self loop
                    weight=0,
                    nextstate=k,
                ),
            )

    return fst


def add_one(fst: kaldifst.StdVectorFst, treat_ilabel_zero_specially: bool):
    """For every non-zero output label, it is increased by one.

    If treat_ilabel_zero_specially is True, then every non-zero input label
    is increased by one. If treat_ilabel_zero_specially is False, then every
    input label is increased by one.

    The input fst is modified in-place.
    """
    for state in kaldifst.StateIterator(fst):
        for arc in kaldifst.ArcIterator(fst, state):
            if treat_ilabel_zero_specially is False or arc.ilabel != 0:
                arc.ilabel += 1

            if arc.olabel != 0:
                arc.olabel += 1

    if fst.input_symbols is not None:
        input_symbols = kaldifst.SymbolTable()
        input_symbols.add_symbol(symbol="<eps>", key=0)

        for i in range(0, fst.input_symbols.num_symbols()):
            s = fst.input_symbols.find(i)
            input_symbols.add_symbol(symbol=s, key=i + 1)

        fst.input_symbols = input_symbols

    if fst.output_symbols is not None:
        output_symbols = kaldifst.SymbolTable()
        output_symbols.add_symbol(symbol="<eps>", key=0)

        for i in range(0, fst.output_symbols.num_symbols()):
            s = fst.output_symbols.find(i)
            output_symbols.add_symbol(symbol=s, key=i + 1)

        fst.output_symbols = output_symbols


def add_disambig_self_loops(fst: kaldifst.StdVectorFst, start: int, end: int):
    """Add self-loops to each state.

    For each disambig symbol, we add a self-loop with input label 0 and output
    label diambig_id of that disambig symbol.

    Args:
      fst:
        It is changed in-place.
      start:
        The ID of #0
      end:
        The ID of the last disambig symbol. For instance if there are 3
        disambig symbols ``#0``, ``#1``, and ``#2``, then ``end`` is the ID
        of ``#3``.
    """
    for state in kaldifst.StateIterator(fst):
        for i in range(start, end + 1):
            fst.add_arc(
                state=state,
                arc=kaldifst.StdArc(
                    ilabel=0,
                    olabel=i,  # if i==k, it is a self loop
                    weight=0,
                    nextstate=state,
                ),
            )

    if fst.output_symbols:
        for i in range(start, end + 1):
            fst.output_symbols.add_symbol(symbol=f"#{i-start}", key=i)
