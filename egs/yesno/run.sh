#!/usr/bin/env bash
# yesno recipe, mirroring ../kaldi/egs/yesno/s5/run.sh (stages, outputs and
# console logs).  Each stage calls this repo's Python API (kaldi_hmm_gmm),
# not kaldi binaries.

set -eou pipefail

stage=-1
stop_stage=100

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) stage=$2; shift 2 ;;
    --stop-stage) stop_stage=$2; shift 2 ;;
    *) echo "$0: unknown option $1" >&2; exit 1 ;;
  esac
done

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  ./local/prepare_data.sh
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  ./prepare_lang.py --lang-dir data/lang
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  ./local/compute_feats.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "steps/train_mono.sh --nj 1 --cmd utils/run.pl --totgauss 400 data/train_yesno data/lang exp/mono0a"
  ./train.py --data-dir data/train_yesno --lang-dir data/lang \
    --feats-scp data/train_yesno/feats.scp --exp-dir exp/mono0a
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ./ali.py --data-dir data/train_yesno --lang-dir data/lang \
    --feats-scp data/train_yesno/feats.scp --exp-dir exp/mono0a
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "utils/mkgraph.sh data/lang_test_tg exp/mono0a exp/mono0a/graph_tgpr"
  ./mkgraph.py --lang-dir data/lang_test_tg --exp-dir exp/mono0a
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "steps/decode.sh --nj 1 --cmd utils/run.pl exp/mono0a/graph_tgpr data/test_yesno exp/mono0a/decode_test_yesno"
  ./decode.py --data-dir data/test_yesno --lang-dir data/lang \
    --feats-scp data/test_yesno/feats.scp --exp-dir exp/mono0a
fi

echo "Done."
