#!/usr/bin/env python3

"""
This scripts generates lexicon.txt inside data/lang_char
"""

from pathlib import Path


def main():
    d = Path("data/lang_char")
    d.mkdir(parents=True, exist_ok=True)

    words = ["zero", "one", "two", "three"]
    words += ["four", "five", "six", "seven"]
    words += ["eight", "nine"]

    with open(d / "lexicon.txt", "w") as f:
        f.write("<SIL> SIL\n")
        for word in words:
            f.write(word)
            f.write(" ")
            f.write(f"{' '.join(list(word))}")
            f.write("\n")


if __name__ == "__main__":
    main()
