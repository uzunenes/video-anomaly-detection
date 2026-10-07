#!/bin/sh
# Download UCSD Ped1/Ped2 and CUHK Avenue and build the frame caches under $VAD_CACHE (default ./cache).
set -e
C=${VAD_CACHE:-cache}
mkdir -p data/avenue "$C"
cd data
[ -f UCSD_Anomaly_Dataset.tar.gz ] || wget -q http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
cd avenue
[ -f Avenue_Dataset.zip ] || wget -q http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
[ -f ground_truth_demo.zip ] || wget -q http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip
cd ../..
python prepare_ucsd.py --root data/UCSD_Anomaly_Dataset.v1p2 --out "$C"
python prepare_avenue.py --zip data/avenue/Avenue_Dataset.zip --gt-zip data/avenue/ground_truth_demo.zip --out "$C/avenue"
