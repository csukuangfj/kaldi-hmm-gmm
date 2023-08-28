#!/usr/bin/env bash

export PYTHONPATH=$PWD/scripts/:$PYTHONPATH

set -eou pipefail

stage=-1
stop_stage=100

dl_dir=$PWD/download

lang_dir=data/lang_phone

. scripts/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"
  mkdir -p $dl_dir

  if [ ! -f $dl_dir/waves_yesno/.completed ]; then
    lhotse download yesno $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare yesno manifest"
  mkdir -p data/manifests
  lhotse prepare yesno $dl_dir/waves_yesno data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for yesno"
  mkdir -p data/fbank
  ./local/compute_fbank_yesno.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Training"
  ./train.py
fi
