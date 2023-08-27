#!/usr/bin/env bash
#
set -eou pipefail

stage=-1
stop_stage=100

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare manifests"
  if [ ! -f ./data/manifests/.done ]; then
    mkdir -p ./data/manifests
    lhotse prepare audio-mnist ~/open-source/AudioMNIST ./data/manifests
    touch ./data/manifests/.done
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Compute fbank"
  if [ ! -f ./data/fbank/.done ]; then
    mkdir -p ./data/fbank
    ./compute_fbank_audio_mnist.py
    touch ./data/fbank/.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Generate lexicon"
  if [ ! -f ./data/lang_char/lexicon.txt ]; then
    mkdir -p ./data/lang_char
    ./generate_lexicon.py
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare lang"
  if [ ! -f ./data/lang_char/.done ]; then
    mkdir -p ./data/lang_char
    ./prepare_lang.py
    touch ./data/lang_char/.done
  fi
fi
