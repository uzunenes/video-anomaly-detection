#!/bin/sh
# Vision-language-model comparison (needs scripts/reproduce.sh first and a GPU with >= 20 GB for the 8B model).
set -e
C=${VAD_CACHE:-cache}
[ -f "$C/avenue_rgb/gt.json" ] || python prepare_avenue.py --zip data/avenue/Avenue_Dataset.zip \
    --gt-zip data/avenue/ground_truth_demo.zip --out "$C/avenue_rgb" --size 360x640 --color
for m in 4B; do
  for spec in "ped2 ped2" "avenue_rgb avenue" "ped1 ped1"; do
    set -- $spec; d=$1; ds=$2
    out=runs/vlm/${d}_qwen3vl_${m}
    [ -f $out/timing.json ] || python tools/vlm_score.py --data "$C/$d" --model Qwen/Qwen3-VL-${m}-Instruct --out $out --batch 16
    python tools/vlm_eval.py --ds $ds --vlm $out --scores . --out $out/eval.json
  done
done
