#!/usr/bin/env bash
# Data preparation for the yesno recipe (mirrors ../kaldi/egs/yesno/s5/local/
# prepare_data.sh + create_yesno_*.pl; same outputs):
#
#   data/local/{waves_all.list,waves.train,waves.test,
#               train_yesno.txt,test_yesno.txt,
#               train_yesno_wav.scp,test_yesno_wav.scp,lm_tg.arpa}
#   data/{train,test}_yesno/{wav.scp,text,utt2spk,spk2utt}
#
# The corpus is downloaded if the waves dir is missing.  Original source
# (what ../kaldi/egs/yesno/s5/run.sh uses): openslr 1,
#   http://www.openslr.org/resources/1/waves_yesno.tar.gz
# We fetch the same tarball from a github.com release mirror to reduce CI
# download time:
#   https://github.com/csukuangfj/kaldi-hmm-gmm/releases/download/v1.1.4/waves_yesno.tar.gz
# The train/test split reproduces create_yesno_waves_test_train.pl: sorted
# listing of the waves dir; first half = train, rest = test; *.wav only.

set -eo pipefail

waves_dir=waves_yesno

while [[ $# -gt 0 ]]; do
  case $1 in
    --waves-dir) waves_dir=$2; shift 2 ;;
    *) echo "$0: unknown option $1" >&2; exit 1 ;;
  esac
done

if [ ! -d "$waves_dir" ]; then
  echo "$0: downloading waves_yesno corpus (github.com mirror of openslr 1)"
  tarball=waves_yesno.tar.gz
  url=https://github.com/csukuangfj/kaldi-hmm-gmm/releases/download/v1.1.4/waves_yesno.tar.gz
  if command -v wget > /dev/null 2>&1; then
    wget -O "$tarball" "$url" || exit 1
  else
    curl -L -o "$tarball" "$url" || exit 1
  fi
  tar -xzf "$tarball" || exit 1
  rm -f "$tarball"
fi

echo "Preparing train and test data"

python3 - "$waves_dir" <<'PYEOF'
import sys
from pathlib import Path

waves_dir = Path(sys.argv[1])
if not waves_dir.is_dir():
    raise SystemExit(f"{waves_dir} does not exist")

local = Path("data/local")
for part in ("train", "test"):
    (Path("data") / f"{part}_yesno").mkdir(parents=True, exist_ok=True)
local.mkdir(parents=True, exist_ok=True)

entries = sorted(p.name for p in waves_dir.iterdir())
n = len(entries)
train, test = [], []
for i, name in enumerate(entries, 1):
    if not name.endswith(".wav"):
        continue
    (train if i <= n / 2 else test).append(name)
def wr(path, s):
    # kaldi-style text artifacts must use LF on every platform (Windows
    # text-mode writes would translate \n to \r\n)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)

wr(local / "waves_all.list", "\n".join(entries) + "\n")
wr(local / "waves.train", "\n".join(train) + "\n")
wr(local / "waves.test", "\n".join(test) + "\n")

def utt_id(name):
    return name[: -len(".wav")]

def transcript(utt):
    return utt.replace("0", "NO").replace("1", "YES").replace("_", " ")

for part, names in (("train", train), ("test", test)):
    u2s = Path(f"data/{part}_yesno")
    txt = "".join(f"{utt_id(x)} {transcript(utt_id(x))}\n" for x in names)
    scp = "".join(f"{utt_id(x)} {waves_dir / x}\n" for x in names)
    wr(local / f"{part}_yesno.txt", txt)
    wr(local / f"{part}_yesno_wav.scp", scp)
    wr(u2s / "text", txt)
    wr(u2s / "wav.scp", scp)
    wr(u2s / "utt2spk", "".join(f"{utt_id(x)} global\n" for x in names))
    wr(u2s / "spk2utt", "global " + " ".join(utt_id(x) for x in names) + "\n")
    print(f"{part}: {len(names)} utterances in {u2s}")

wr(Path("data/local/lm_tg.arpa"), Path("input/task.arpabo").read_text())
print("wrote data/local/lm_tg.arpa")
PYEOF
echo "$0: done."
