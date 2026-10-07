#!/bin/sh
# Final method on Ped2, Ped1 and Avenue: 2d AEAN without adversarial training (intensity channel I2) plus the
# DIS-medium optical-flow speed (Fsm) and direction (Fdm) channels, fused with Fisher's method.
# The evaluation tags are the ones tools/fuse_runs.py reads (I2: pool48_mc4, Fsm/Fdm: pool48_mc5).
set -e
C=${VAD_CACHE:-cache}
for spec in "ped2 1" "ped1 1" "avenue 2"; do        # flow frame gap: UCSD 10 fps, Avenue 25 fps
  set -- $spec; ds=$1; gap=$2
  run=runs/${ds}_2d_noadv
  [ -f $run/model.pt ] || python train.py --data "$C/$ds" --variant 2d --no-adv --out $run
  python evaluate.py --data "$C/$ds" --run $run --pool 48 --no-state --channels "" --tag pool48_mc4
  python evaluate.py --data "$C/$ds" --run $run --pool 48 --no-state --channels flowm,flowrxm --flow-gap $gap \
      --workers 8 --tag pool48_mc5
done
CHANNELS=I2,Fsm,Fdm python tools/temporal_protocols.py ped2 ped1 avenue
