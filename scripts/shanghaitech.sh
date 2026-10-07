#!/bin/sh
# ShanghaiTech Campus (12 test scenes) with the final method, one model per scene, and the protocol table.
# $1 = folder holding the official split archive shanghaitech.tar.gz.a? (or a zip with the same layout)
set -e
C=${VAD_CACHE:-cache}
if [ ! -f "$C/shtech_01/gt.json" ]; then
  if [ -d "$1" ]; then
    [ -d data/shanghaitech/testing ] || { mkdir -p data && cat "$1"/shanghaitech.tar.gz.a? | tar xz -C data; }
    python prepare_shanghaitech.py --root data/shanghaitech --out "$C"
  else
    python prepare_shanghaitech.py --zip "$1" --out "$C"
  fi
fi
for d in $(ls "$C"/shtech_*/gt.json | xargs -n1 dirname | sort); do          # scene 13 has no test videos
  sc=${d##*_}; run=runs/shtech_${sc}_2d_noadv
  [ -f $run/model.pt ] || python train.py --data $d --variant 2d --no-adv --out $run
  python evaluate.py --data $d --run $run --pool 48 --tag pool48_mc5 --no-state --channels flowm,flowrxm --flow-gap 2 --workers 8
done
python tools/shtech_protocols.py --runs runs --cache "$C"
