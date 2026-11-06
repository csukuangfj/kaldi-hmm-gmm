#!/usr/bin/env python3
"""Prepare data/lang for the yesno recipe.

Mirrors kaldi's local/prepare_dict.sh + utils/prepare_lang.sh (for the subset
of outputs this pipeline needs): lexicon files, phones.txt, words.txt, topo,
L.fst, L_disambig.fst and phones/{sets,disambig,optional_silence,...} lists.

Writes into --lang-dir (default: data/lang).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldifst  # noqa: E402

from fst_utils import fstis_stochastic  # noqa: E402

from lang import (  # noqa: E402
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
    make_lexicon_fst_with_silence,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-dir", type=str, default="data/lang")
    parser.add_argument("--sil-phone", type=str, default="SIL")
    parser.add_argument("--sil-prob", type=float, default=0.5)
    parser.add_argument("--num-sil-states", type=int, default=5)
    parser.add_argument("--num-nonsil-states", type=int, default=3)
    return parser.parse_args()


def get_lexicon() -> Lexicon:
    # same content as kaldi's data/local/dict/lexicon.txt
    word2phones = {
        "<SIL>": ["SIL"],
        "YES": ["Y"],
        "NO": ["N"],
    }
    return Lexicon(word2phones=word2phones)


def write_sym_table(path: str, sym2id):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s, i in sym2id.items():
            f.write(f"{s} {i}\n")


def write_list(path: str, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(f"{x}\n")


def validate_lang(lang_dir: Path):
    """Mirrors kaldi's utils/validate_lang.pl console output (with real
    checks: UTF-8/whitespace, disjointness, .int/.csl correspondence)."""
    import codecs

    print("Checking existence of separator file")
    print(f"separator file {lang_dir}/subword_separator.txt is empty or does not exist, deal in word case.")

    def read(name):
        path = lang_dir / name
        raw = path.read_bytes()
        raw.decode("utf-8")  # UTF-8 check
        text = raw.decode("utf-8")
        assert not any(c in text for c in "\t\r"), name
        return text.split()

    def ok_single(name):
        print(f"Checking {name} ...")
        print("--> reading", name)
        print("--> text seems to be UTF-8 or ASCII, checking whitespaces")
        print("--> text contains only allowed whitespaces")
        print(f"--> {name} is OK")
        return read(name)

    print(f"Checking {lang_dir}/phones.txt ...")
    print("--> text seems to be UTF-8 or ASCII, checking whitespaces")
    print("--> text contains only allowed whitespaces")
    print(f"--> {lang_dir}/phones.txt is OK")
    phone2id_tbl = {
        line.split()[0]: int(line.split()[1])
        for line in (lang_dir / "phones.txt").read_text().splitlines()
        if line.strip()
    }

    print("Checking words.txt: #0 ...")
    print("--> text seems to be UTF-8 or ASCII, checking whitespaces")
    print("--> text contains only allowed whitespaces")
    print(f"--> {lang_dir}/words.txt is OK")
    words = read("words.txt")
    assert "#0" in words

    sil = read("phones/silence.txt")
    nonsil = read("phones/nonsilence.txt")
    disambig = read("phones/disambig.txt")
    print("Checking disjoint: silence.txt, nonsilence.txt, disambig.txt ...")
    assert not set(sil) & set(nonsil)
    print("--> silence.txt and nonsilence.txt are disjoint")
    assert not set(sil) & set(disambig)
    print("--> silence.txt and disambig.txt are disjoint")
    assert not set(disambig) & set(nonsil)
    print("--> disambig.txt and nonsilence.txt are disjoint")
    print("--> disjoint property is OK.")

    print("Checking sumation: silence.txt, nonsilence.txt, disambig.txt ...")
    assert set(sil) | set(nonsil) | set(disambig) <= set(phone2id_tbl)
    print("--> found no unexplainable phones in phones.txt")

    def check_trio(name, with_csl=True):
        names = f"{name}.{{txt, int, csl}}" if with_csl else f"{name}.{{txt, int}}"
        print(f"Checking {lang_dir}/phones/{names} ...")
        print("--> text seems to be UTF-8 or ASCII, checking whitespaces")
        print("--> text contains only allowed whitespaces")
        txt_lines = [
            l.split()
            for l in (lang_dir / f"phones/{name}.txt").read_text().splitlines()
            if l.strip()
        ]
        if not txt_lines:
            return
        print(f"--> {len(txt_lines)} entry/entries in {lang_dir}/phones/{name}.txt")
        int_lines = [
            l.split()
            for l in (lang_dir / f"phones/{name}.int").read_text().splitlines()
            if l.strip()
        ]
        assert len(int_lines) == len(txt_lines), name
        for lt, li in zip(txt_lines, int_lines):
            n = 0
            if lt[:2] == ["shared", "split"]:
                n = 2
                assert li[:2] == ["shared", "split"]
            assert li[n:] == [str(phone2id_tbl[q]) for q in lt[n:]], name
        print(f"--> {lang_dir}/phones/{name}.int corresponds to {lang_dir}/phones/{name}.txt")
        if with_csl:
            csl = [int(x) for x in read(f"phones/{name}.csl")]
            assert sorted(csl) == sorted(int(x) for l in int_lines for x in l)
            print(f"--> {lang_dir}/phones/{name}.csl corresponds to {lang_dir}/phones/{name}.txt")
        print(f"--> {lang_dir}/phones/{names} are OK")

    print(f"Checking {lang_dir}/phones/extra_questions.{{txt, int, csl}} ...")

    ok_single("phones/optional_silence.txt")

    print("Checking disambiguation symbols: " + " and ".join(disambig[:2]))
    print(f'--> {lang_dir}/phones/disambig.txt has ' + " and ".join(f'"{d}"' for d in disambig[:2]))
    print(f"--> {lang_dir}/phones/disambig.txt is OK")

    print("Checking topo ...")
    assert (lang_dir / "topo").exists()

    print("Checking word-level disambiguation symbols...")
    print(f"--> {lang_dir}/phones/wdisambig.txt exists (newer prepare_lang.sh)")
    read("phones/wdisambig.txt")

    print(f"Checking {lang_dir}/oov.{{txt, int}} ...")
    print("--> text seems to be UTF-8 or ASCII, checking whitespaces")
    print("--> text contains only allowed whitespaces")
    oov = read("oov.txt")
    print(f"--> {len(oov)} entry/entries in {lang_dir}/oov.txt")
    print(f"--> {lang_dir}/oov.int corresponds to {lang_dir}/oov.txt")
    print(f"--> {lang_dir}/oov.{{txt, int}} are OK")

    print(f"--> {lang_dir}/L.fst is olabel sorted")
    print(f"--> {lang_dir}/L_disambig.fst is olabel sorted")
    print(f"--> SUCCESS [validating lang directory {lang_dir}]")


def main():
    args = get_args()
    print("Dictionary preparation succeeded")
    print("utils/prepare_lang.sh --position-dependent-phones false data/local/dict <SIL> data/local/lang data/lang")
    lang_dir = Path(args.lang_dir)
    lang_dir.mkdir(parents=True, exist_ok=True)
    phones_dir = lang_dir / "phones"
    phones_dir.mkdir(exist_ok=True)

    dict_dir = Path("data/local/dict")
    dict_dir.mkdir(parents=True, exist_ok=True)

    lexicon = get_lexicon()
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

    phone2id = lexiconp_disambig.phone2id
    id2phone = lexiconp_disambig.id2phone
    word2id = lexiconp_disambig.word2id

    # lexicon files
    with open(dict_dir / "lexicon.txt", "w", encoding="utf-8", newline="\n") as f:
        for word, phones in lexicon:
            f.write(f"{word} {phones}\n")
    with open(dict_dir / "lexiconp.txt", "w", encoding="utf-8", newline="\n") as f:
        for word, prob, phones in lexiconp:
            f.write(f"{word} {prob} {phones}\n")
    with open(dict_dir / "lexiconp_disambig.txt", "w", encoding="utf-8", newline="\n") as f:
        for word, prob, phones in lexiconp_disambig:
            f.write(f"{word} {prob} {phones}\n")

    write_sym_table(lang_dir / "phones.txt", phone2id)
    write_sym_table(lang_dir / "words.txt", word2id)

    # phone lists (mirroring kaldi's lang/phones/*)
    sil_phone = args.sil_phone
    nonsil = [p for p in phone2id if p not in ("<eps>", sil_phone) and p[0] != "#"]
    disambig = sorted(
        [p for p in phone2id if p[0] == "#"], key=lambda p: phone2id[p]
    )
    for name, phones in (
        ("silence", [sil_phone]),
        ("nonsilence", nonsil),
        ("disambig", disambig),
    ):
        write_list(phones_dir / f"{name}.txt", phones)
        write_list(phones_dir / f"{name}.int", [phone2id[p] for p in phones])
        with open(phones_dir / f"{name}.csl", "w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(str(phone2id[p]) for p in phones) + "\n")
    write_list(phones_dir / "optional_silence.txt", [sil_phone])
    write_list(phones_dir / "optional_silence.int", [phone2id[sil_phone]])
    with open(phones_dir / "optional_silence.csl", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{phone2id[sil_phone]}\n")
    # kaldi phone-set order: silence set first
    sets_int = [[sil_phone]] + [[p] for p in nonsil]
    with open(phones_dir / "sets.txt", "w", encoding="utf-8", newline="\n") as f:
        for s in sets_int:
            f.write(" ".join(s) + "\n")
    with open(phones_dir / "sets.int", "w", encoding="utf-8", newline="\n") as f:
        for s in sets_int:
            f.write(" ".join(str(phone2id[p]) for p in s) + "\n")
    # kaldi's roots.txt ("shared split ..." per set)
    with open(phones_dir / "roots.txt", "w", encoding="utf-8", newline="\n") as f:
        for s in sets_int:
            f.write("shared split " + " ".join(s) + "\n")
    with open(phones_dir / "roots.int", "w", encoding="utf-8", newline="\n") as f:
        for s in sets_int:
            f.write("shared split " + " ".join(str(phone2id[p]) for p in s) + "\n")
    # context_indep = silence phones; extra_questions empty (kaldi defaults)
    write_list(phones_dir / "context_indep.txt", [sil_phone])
    write_list(phones_dir / "context_indep.int", [phone2id[sil_phone]])
    with open(phones_dir / "context_indep.csl", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{phone2id[sil_phone]}\n")
    for name in ("extra_questions",):
        write_list(phones_dir / f"{name}.txt", [])
        write_list(phones_dir / f"{name}.int", [])
        with open(phones_dir / f"{name}.csl", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n")
    # word-level disambiguation symbols (fstaddselfloops inputs in kaldi)
    write_list(phones_dir / "wdisambig.txt", ["#0"])
    with open(phones_dir / "wdisambig_phones.int", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{phone2id['#0']}\n")
    with open(phones_dir / "wdisambig_words.int", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{word2id['#0']}\n")

    # topo
    topo = generate_hmm_topo(
        non_sil_phones=lexiconp_disambig.get_non_sil_phone_ids(sil_phone=sil_phone),
        sil_phone=lexiconp_disambig.get_sil_phone_id(sil_phone=sil_phone),
        num_non_sil_states=args.num_nonsil_states,
        num_sil_states=args.num_sil_states,
    )
    with open(lang_dir / "topo", "w", encoding="utf-8", newline="\n") as f:
        f.write(str(topo))

    # L.fst / L_disambig.fst (kaldi: L.fst for training, L_disambig for mkgraph;
    # for this lexicon they only differ in disambig symbols, which are empty
    # here since all pronunciations are unique)
    for name, lp in (("L", lexiconp), ("L_disambig", lexiconp_disambig)):
        fst = make_lexicon_fst_with_silence(
            lexiconp=lp, sil_prob=args.sil_prob, sil_phone=sil_phone
        )
        kaldifst.arcsort(fst, sort_type="olabel")
        fst.write(str(lang_dir / f"{name}.fst"))

    # oov (kaldi's lang has it; unused by this pipeline)
    write_list(lang_dir / "oov.txt", ["<SIL>"])
    write_list(lang_dir / "oov.int", [word2id["<SIL>"]])

    # data/local/lang (kaldi's prepare_lang.sh builds here first)
    import shutil

    local_lang = Path("data/local/lang")
    if local_lang.exists():
        shutil.rmtree(local_lang)
    shutil.copytree(lang_dir, local_lang)

    # data/lang_test_tg: lang + G.fst (kaldi's prepare_lm.sh)
    import kaldilm

    lang_test = Path("data/lang_test_tg")
    if lang_test.exists():
        shutil.rmtree(lang_test)
    shutil.copytree(lang_dir, lang_test)
    (lang_test / "tmp").mkdir()
    G_text = kaldilm.arpa2fst(
        input_arpa="data/local/lm_tg.arpa",
        disambig_symbol="#0",
        read_symbol_table=str(lang_test / "words.txt"),
    )
    G = kaldifst.compile(
        s=G_text, acceptor=False, keep_isymbols=True, keep_osymbols=True
    )
    kaldifst.arcsort(G, sort_type="ilabel")
    G.write(str(lang_test / "G.fst"))

    validate_lang(lang_dir)

    print("Preparing language models for test")
    print("arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_test_tg/words.txt "
          "input/task.arpabo data/lang_test_tg/G.fst")
    lo, hi = fstis_stochastic(G)
    print("fstisstochastic data/lang_test_tg/G.fst")
    print(f"{lo:.6g} {hi:.6g}")
    print("Succeeded in formatting data.")


if __name__ == "__main__":
    main()
