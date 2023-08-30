#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

"""
The lang_dir should contain a file "lexicon.txt"

See http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating

---

steps/train_mono.sh requires the following files:
    - phones.txt
    - phones/sets.txt
    - phones/disambig.txt
    - L.fst
    - words.txt
"""

import argparse
import copy
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kaldi_hmm_gmm as khg
import kaldifst


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=Path,
        required=True,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        Generated files by this script are saved into this directory.
        """,
    )

    parser.add_argument(
        "--num-sil-states",
        type=int,
        default=5,
        help="Number of states of silence phones in the HMM topology",
    )
    parser.add_argument(
        "--num-nonsil-states",
        type=int,
        default=3,
        help="Number of states of non-silence phones in the HMM topology",
    )

    parser.add_argument(
        "--sil-phone",
        type=str,
        default="SIL",
        help="Silence phone. Must exist in lexicon.txt",
    )

    return parser.parse_args()


class Lexicon:
    """Once constructed it is immutable"""

    def __init__(
        self,
        lexicon_txt: Optional[str] = None,
        word2phones: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
          lexicon_txt:
            Path to lexicon.txt. Each line in the lexicon.txt has the following
            format:

                word phone1 phone2 ... phoneN

            That is, the first field is the word, the remaining fields are
            pronunciations of this word. Fields are separated by space(s).
          word2phones:
            If provided, lexicon_txt is ignored. it maps a word to a list
            of phones. It is a list since a word may have multiple pronunciations
        """
        if word2phones is not None:
            self.word2phones = copy.deepcopy(word2phones)
            return

        word2phones = defaultdict(list)
        with open(lexicon_txt, encoding="utf-8") as f:
            for line in f:
                word_phones = line.strip().split()
                assert len(word_phones) >= 2, (word_phones, line)
                word = word_phones[0]
                phones = " ".join(word_phones[1:])
                word2phones[word].append(phones)

        self.word2phones = word2phones

    def __iter__(self) -> Tuple[str, List[str]]:
        for word, phones_list in self.word2phones.items():
            for phones in phones_list:
                yield word, phones

    def __str__(self):
        return str(self.word2phones)

    @staticmethod
    def from_lexiconp(lexiconp: "Lexiconp"):
        """Convert a Lexiconp to a Lexicon"""
        word2phones = defaultdict(list)
        for word, _, phones in lexiconp:
            # We don't copy phones here!
            word2phones[word].append(phones)
        return Lexicon(word2phones=word2phones)


class Lexiconp:
    """Once constructed it is immutable"""

    def __init__(
        self,
        lexiconp_txt: Optional[str] = None,
        word2prob_phones: Optional[dict] = None,
    ):
        """
        Args:
          lexicon_txt:
            Path to lexiconp.txt. Each line in the lexiconp.txt has the following
            format:

              word prob phone1 phone2 ... phoneN

            That is, the first field is the word, the remaining fields are
            pronunciations of this word. Fields are separated by space(s).
          word2prob_phones:
            If provided, `lexiconp_txt` is ignored.
        """
        if word2prob_phones is not None:
            self.word2prob_phones = copy.deepcopy(word2prob_phones)
            return

        # Each entry in word_phones_list is a list containing:
        #  [word, prob, "phone1 phone2 ... phoneN"]
        word2prob_phones = defaultdict(list)
        with open(lexiconp_txt, encoding="utf-8") as f:
            for line in f:
                word_prob_phones = line.strip().split()
                assert len(word_prob_phones) >= 3, (word_prob_phones, line)
                word = word_prob_phones[0]
                prob = word_prob_phones[1]
                phones = " ".join(word_prob_phones[2:])
                word2prob_phones[word].append([prob, phones])

        self.word2prob_phones = word2prob_phones

    def __iter__(self):
        for word, prob_phones_list in self.word2prob_phones.items():
            for prob, phones in prob_phones_list:
                yield word, prob, phones

    def __str__(self):
        return str(self.word2prob_phones)

    @staticmethod
    def from_lexicon(lexicon: Lexicon):
        """Convert a Lexicon to a Lexiconp"""
        word2prob_phones = defaultdict(list)
        for word, phones in lexicon:
            word2prob_phones[word].append(["1.0", phones])

        return Lexiconp(word2prob_phones=word2prob_phones)

    def add_lex_disambig(self) -> "Lexiconp":
        """See also add_lex_disambig.pl from kaldi."""
        # (1) Work out the count of each phone-sequence in the lexicon.
        count = defaultdict(int)
        for _, _, phones in self:
            count[phones] += 1

        # (2) For each left sub-sequence of each phone-sequence, note down
        # that it exists (for identifying prefixes of longer strings).
        issubseq = defaultdict(int)
        for _, _, phones in self:
            phones = phones.split()
            phones.pop()
            while phones:
                issubseq[" ".join(phones)] = 1
                phones.pop()

        # (3) For each entry in the lexicon:
        # if the phone sequence is unique and is not a
        # prefix of another word, no disambig symbol.
        # Else output #1, or #2, #3, ... if the same phone-seq
        # has already been assigned a disambig symbol.

        # We start with #1 since #0 has its own purpose
        first_allowed_disambig = 1
        max_disambig = first_allowed_disambig - 1
        last_used_disambig_symbol_of = defaultdict(int)

        disambig_word2prob_phones = defaultdict(list)

        for word, prob, phones in self:
            assert phones != "", phones
            if issubseq[phones] == 0 and count[phones] == 1:
                disambig_word2prob_phones[word].append([prob, phones])
                continue

            cur_disambig = last_used_disambig_symbol_of[phones]
            if cur_disambig == 0:
                cur_disambig = first_allowed_disambig
            else:
                cur_disambig += 1

            if cur_disambig > max_disambig:
                max_disambig = cur_disambig

            last_used_disambig_symbol_of[phones] = cur_disambig
            phones += f" #{cur_disambig}"
            disambig_word2prob_phones[word].append([prob, phones])

        ans = Lexiconp(word2prob_phones=disambig_word2prob_phones)

        # The largest disambig ID we have used. If none was used, then
        # it is 0
        ans._max_disambig = max_disambig

        return ans

    @property
    def phone2id(self):
        if hasattr(self, "_phone2id"):
            return self._phone2id

        if not hasattr(self, "_max_disambig"):
            self._max_disambig = 0

        phone_set = set()
        for _, _, phones in self:
            phone_set.update(phones.split())
        phone_list = list(phone_set)
        kept_phone_list = []

        for p in phone_list:
            if p[0] == "#":
                continue
            kept_phone_list.append(p)

        kept_phone_list.sort()
        kept_phone_list.remove("SIL")
        kept_phone_list.insert(0, "<eps>")
        kept_phone_list.insert(1, "SIL")
        for i in range(self._max_disambig + 2):
            kept_phone_list.append(f"#{i}")

        self._phone2id = {p: i for i, p in enumerate(kept_phone_list)}
        self._id2phone = {i: p for i, p in enumerate(kept_phone_list)}
        return self._phone2id

    @property
    def id2phone(self):
        if hasattr(self, "_id2phone"):
            return self._id2phone

        _ = self.phone2id

        return self._id2phone

    @property
    def word2id(self):
        if hasattr(self, "_word2id"):
            return self._word2id

        word_list = list(self.word2prob_phones.keys())
        word_list.sort()
        word_list.insert(0, "<eps>")
        word_list.append("#0")
        word_list.append("<s>")
        word_list.append("</s>")

        self._word2id = {w: i for i, w in enumerate(word_list)}
        self._id2word = {i: w for i, w in enumerate(word_list)}

        assert len(word_list) == len(self._word2id), (
            len(word_list),
            len(self._word2id),
        )

        return self._word2id

    @property
    def id2word(self):
        if hasattr(self, "_id2word"):
            return self._id2word

        _ = self.word2id

        return self._id2word

    def get_non_sil_phone_ids(self, sil_phone: str = "SIL") -> List[int]:
        skip = ("<eps>", sil_phone)
        ans = []
        for p, i in self.phone2id.items():
            if p in skip or p[0] == "#":
                # disambig symbols are #0, #1, #2, ..., #N
                continue
            ans.append(i)

        return ans

    def get_sil_phone_id(self, sil_phone: str = "SIL") -> int:
        return self.phone2id[sil_phone]


# See also
# http://vpanayotov.blogspot.com/2012/06/kaldi-decoding-graph-construction.html
def make_lexicon_fst_with_silence(
    lexiconp: Lexiconp,
    sil_prob: float = 0.5,
    sil_phone: str = "SIL",
    sil_disambig: Optional[int] = None,
) -> kaldifst.StdVectorFst:
    phone2id = lexiconp.phone2id
    word2id = lexiconp.word2id

    assert sil_phone in phone2id, sil_phone

    sil_cost = -1 * math.log(sil_prob)
    no_sil_cost = -1 * math.log(1.0 - sil_prob)

    fst = kaldifst.StdVectorFst()

    start_state = fst.add_state()
    loop_state = fst.add_state()
    sil_state = fst.add_state()

    fst.start = start_state
    fst.set_final(state=loop_state, weight=0)

    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=0,
            olabel=0,
            weight=no_sil_cost,
            nextstate=loop_state,
        ),
    )

    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=0,
            olabel=0,
            weight=sil_cost,
            nextstate=sil_state,
        ),
    )

    if sil_disambig is None:
        fst.add_arc(
            state=sil_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[sil_phone],
                olabel=0,
                weight=0,
                nextstate=loop_state,
            ),
        )
    else:
        sil_disambig_state = fst.add_state()
        fst.add_arc(
            state=sil_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[sil_phone],
                olabel=0,
                weight=0,
                nextstate=sil_disambig_state,
            ),
        )

        fst.add_arc(
            state=sil_disambig_state,
            arc=kaldifst.StdArc(
                ilabel=sil_disambig,
                olabel=0,
                weight=0,
                nextstate=loop_state,
            ),
        )

    for word, prob, phones in lexiconp:
        phoneseq = phones.split()
        pron_cost = -1 * math.log(float(prob))
        cur_state = loop_state

        for i in range(len(phoneseq) - 1):
            next_state = fst.add_state()
            fst.add_arc(
                state=cur_state,
                arc=kaldifst.StdArc(
                    ilabel=phone2id[phoneseq[i]],
                    olabel=word2id[word] if i == 0 else 0,
                    weight=pron_cost if i == 0 else 0,
                    nextstate=next_state,
                ),
            )
            cur_state = next_state

        i = len(phoneseq) - 1  # note: i == -1 if phoneseq is empty.

        fst.add_arc(
            state=cur_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[phoneseq[i]] if i >= 0 else 0,
                olabel=word2id[word] if i <= 0 else 0,
                weight=no_sil_cost + (pron_cost if i <= 0 else 0),
                nextstate=loop_state,
            ),
        )

        fst.add_arc(
            state=cur_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[phoneseq[i]] if i >= 0 else 0,
                olabel=word2id[word] if i <= 0 else 0,
                weight=sil_cost + (pron_cost if i <= 0 else 0),
                nextstate=sil_state,
            ),
        )

    # attach symbol table
    isym = kaldifst.SymbolTable()
    for p, i in phone2id.items():
        isym.add_symbol(symbol=p, key=i)
    fst.input_symbols = isym

    osym = kaldifst.SymbolTable()
    for w, i in word2id.items():
        osym.add_symbol(symbol=w, key=i)
    fst.output_symbols = osym

    return fst


def generate_hmm_topo(
    non_sil_phones: List[int],
    sil_phone: int,
    num_non_sil_states: int = 3,
    num_sil_states: int = 5,
) -> khg.HmmTopology:
    """
    Args:
      num_non_sil_states:
        Number of emitting states for non-silence phones.
      num_sil_states:
        Number of emitting states for silence phones.
      non_sil_phones:
        List of non-silence phones.
      sil_phone:
        The silence phone.
    """
    s = ""
    s += "<Topology> "

    # for nonsil_phone
    s += "<TopologyEntry> "
    s += "<ForPhones> "
    s += " ".join(map(str, non_sil_phones))
    s += "\n</ForPhones> "
    for i in range(num_non_sil_states):
        s += f"<State> {i} <PdfClass> {i} "
        s += f"<Transition> {i} 0.75 "
        s += f"<Transition> {i+1} 0.25 "
        s += "</State> "

    s += f"<State> {num_non_sil_states} </State> "  # non-emitting final state.
    s += "</TopologyEntry> "

    # Now for silence phones.  They have a different topology-- apart from
    # the first and last states, it's fully connected, as long as you
    # have >= 3 states.

    if num_sil_states > 1:
        # Transitions to all but last emitting state.
        transp = 1.0 / (num_sil_states - 1)

        s += "<TopologyEntry> "
        s += "<ForPhones> "
        s += f"{sil_phone} "
        s += "</ForPhones> "

        s += "<State> 0 <PdfClass> 0 "
        for i in range(num_sil_states - 1):
            # Transitions to all but last emitting state.
            s += f"<Transition> {i} {transp} "
        s += "</State> "

        # the central states all have transitions to
        # themselves and to the last emitting state.
        for i in range(1, num_sil_states - 1):
            s += f"<State> {i} <PdfClass> {i} "
            for k in range(1, num_sil_states):
                s += f"<Transition> {k} {transp} "
            s += "</State> "

        # Final emitting state (non-skippable).
        s += f"<State> {num_sil_states - 1} <PdfClass> {num_sil_states - 1} "
        s += f"<Transition> {num_sil_states - 1} 0.75 "
        s += f"<Transition> {num_sil_states} 0.25 "
        s += "</State> "

        # Final nonemitting state:
        s += f"<State> {num_sil_states} </State> "
        s += "</TopologyEntry> "
    else:
        assert num_sil_states == 1, num_sil_states
        s += "<TopologyEntry>"
        s += "<ForPhones>"
        s += f"{sil_phone}"
        s += "</ForPhones>"
        s += "<State> 0 <PdfClass> 0"
        s += "<Transition> 0 0.75"
        s += "<Transition> 1 0.25"
        s += "</State>"
        s += f"<State> {num_sil_states} </State>"
        s += "</TopologyEntry>"

    topo = khg.HmmTopology()
    topo.read(s)

    return topo


def main():
    args = get_args()
    logging.info(vars(args))
    assert (args.lang_dir / "lexicon.txt").is_file()

    lexicon = Lexicon(lexicon_txt=args.lang_dir / "lexicon.txt")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
