#!/usr/bin/env bash

set -ex
lang_dir=data/lang

rm -rf $lang_dir
mkdir -p $lang_dir

cat > $lang_dir/lexicon.txt <<EOF
hell h e l l
he h e
hello h e l l o
hi h i
are a r e
<sil> SIL
EOF

cat > $lang_dir/phones.txt <<EOF
SIL
a
e
h
i
l
o
r
EOF

cat $lang_dir/phones.txt | grep -v SIL > $lang_dir/nonsilence_phones.txt

echo SIL > $lang_dir/silence_phones.txt
echo SIL > $lang_dir/optional_silence.txt
