#!/usr/bin/env python3

"""
The lang_dir should contain the following files:
    - lexicon.txt
    - silence_phones.txt
    - nonsilence_phones.txt
    - optional_silence_phone.txt

It can also contain an optional file "extra_questions.txt" an "lexiconp.txt".
If "lexiconp.txt" exists, we will generate "lexicon.txt" from it even if
"lexicon.txt" exists.

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
import logging
import shutil
from collections import defaultdict
from pathlib import Path


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
        help="states in silence models",
    )
    parser.add_argument(
        "--num-nonsil-states",
        type=int,
        default=3,
        help="#states in non-silence models",
    )

    parser.add_argument(
        "--oov-word",
        type=str,
        default="<SIL>",
        help="#states in non-silence models",
    )

    return parser.parse_args()


def generate_lexicon_from_lexiconp(lexiconp: str, lexicon: str) -> None:
    """Create lexicon.txt from lexiconp.txt"""
    word_phones_list = []
    with open(lexiconp) as f:
        for line in f:
            word_prob_phones = line.strip().split()
            assert len(word_prob_phones) >= 3, (line, word_prob_phones)
            word = word_prob_phones[0]
            phones = " ".join(word_prob_phones[2:])
            word_phones_list.append([word, phones])

    with open(lexicon, "w") as f:
        for word, phones in word_phones_list:
            f.write(f"{word} {phones}\n")


def generate_lexiconp_from_lexicon(lexiconp: str, lexicon: str) -> None:
    """Create lexiconp.txt from lexicon.txt"""
    word_phones_list = []
    with open(lexicon) as f:
        for line in f:
            word_phones = line.strip().split()
            assert len(word_phones) >= 2, (line, word_phones)
            word = word_phones[0]
            phones = " ".join(word_phones[1:])
            word_phones_list.append([word, phones])

    with open(lexiconp, "w") as f:
        for word, phones in word_phones_list:
            f.write(f"{word} 1.0 {phones}\n")


def generate_phone_sets(lang_dir: Path):
    """Generate lang_dir/phones/sets.txt from lang_dir/silence_phones.txt
    and lang_dir/nonsilence_phones.txt.
    """
    (lang_dir / "phones").mkdir(exist_ok=True)

    silence_phones = []
    with open(lang_dir / "silence_phones.txt") as f:
        for line in f:
            p = line.strip().split()
            assert len(p) == 1, (line, p)
            silence_phones.append(p[0])

    nonsilence_phones = []
    with open(lang_dir / "nonsilence_phones.txt") as f:
        for line in f:
            p = line.strip().split()
            assert len(p) == 1, (line, p)
            nonsilence_phones.append(p[0])

    with open(lang_dir / "phones" / "sets.txt", "w") as f:
        for p in silence_phones:
            f.write(f"{p}\n")

        for p in nonsilence_phones:
            f.write(f"{p}\n")


def add_lex_disambig(lexiconp: str, lexiconp_disambig: str) -> int:
    """
    See also add_lex_disambig.pl from kaldi.

    Args:
      lexiconp:
        Path to lexiconp.txt
      lexiconp_disambig:
        File to save the results.
    Returns:
      Return the max disambig symbol that was used. If no disambig symbols
      were used, then it returns 0.
    """

    # (0) read lexiconp.txt first
    word_prob_phones_list = []
    with open(lexiconp) as f:
        for line in f:
            word_prob_phones = line.strip().split()
            assert len(word_prob_phones) >= 3, (line, word_prob_phones)
            word = word_prob_phones[0]
            prob = word_prob_phones[1]
            phones = word_prob_phones[2:]
            word_prob_phones_list.append([word, prob, phones])

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, _, phones in word_prob_phones_list:
        count[" ".join(phones)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, _, phones in word_prob_phones_list:
        phones = phones.copy()
        phones.pop()
        while phones:
            issubseq[" ".join(phones)] = 1
            phones.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    disambig_word_prob_phoneseq_list = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, prob, phones in word_prob_phones_list:
        phoneseq = " ".join(phones)
        assert phoneseq != ""
        if issubseq[phoneseq] == 0 and count[phoneseq] == 1:
            disambig_word_prob_phoneseq_list.append((word, prob, phoneseq))
            continue

        cur_disambig = last_used_disambig_symbol_of[phoneseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig

        last_used_disambig_symbol_of[phoneseq] = cur_disambig
        phoneseq += f" #{cur_disambig}"
        disambig_word_prob_phoneseq_list.append((word, prob, phoneseq))

    with open(lexiconp_disambig, "w") as f:
        for word, prob, phoneseq in disambig_word_prob_phoneseq_list:
            f.write(f"{word} {prob} {phoneseq}\n")

    return max_disambig


def main():
    args = get_args()
    logging.info(vars(args))

    assert args.lang_dir.is_dir()
    if (args.lang_dir / "phones").is_dir():
        logging.info(f"Removing existing directory {args.lang_dir}/phones")
        shutil.rmtree(args.lang_dir / "phones")

    if not (args.lang_dir / "lexicon.txt").is_file():
        assert (args.lang_dir / "lexiconp.txt").is_file(), (
            args.lang_dir / "lexiconp.txt"
        )
        logging.info("Generating lexicon.txt from lexiconp.txt")
        generate_lexicon_from_lexiconp(
            lexiconp=args.lang_dir / "lexiconp.txt",
            lexicon=args.lang_dir / "lexicon.txt",
        )

    if not (args.lang_dir / "lexiconp.txt").is_file():
        assert (args.lang_dir / "lexicon.txt").is_file(), args.lang_dir / "lexicon.txt"
        logging.info("Generating lexiconp.txt from lexicon.txt")
        generate_lexiconp_from_lexicon(
            lexiconp=args.lang_dir / "lexiconp.txt",
            lexicon=args.lang_dir / "lexicon.txt",
        )

    assert (args.lang_dir / "silence_phones.txt").is_file()
    assert (args.lang_dir / "nonsilence_phones.txt").is_file()

    generate_phone_sets(args.lang_dir)
    max_disambig = add_lex_disambig(
        lexiconp=args.lang_dir / "lexiconp.txt",
        lexiconp_disambig=args.lang_dir / "lexiconp_disambig.txt",
    )
    print(max_disambig)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
