# Video Anomaly Detection: label-free, live statistical background modelling for fixed cameras

A lightweight one-class video anomaly detector for static surveillance cameras. It learns only from normal footage of
the camera, runs causally (no future frames, no per-video statistics) and runs in real time on a 4-core CPU.

The method adapts the hyperspectral AEAN + RX anomaly detectors of Arısoy (GTU PhD thesis, 2021) to video. The time
axis takes the place of the spectral axis, and the group's statistical background modelling is applied to
autoencoder residuals and to optical-flow statistics.

## Method

```
frame ─┬─> autoencoder residual (AEAN 2d, 16×16 patches) ──> per-cell FRX-Bayes (RX, Bayesian shrinkage) ─┐
       ├─> DIS optical flow: speed ──────────> GMM-UBM, per-cell MAP adaptation ──────────────────────────┼─> tail probability p
       └─> DIS optical flow: 8-bin direction histogram + max speed ──> per-cell FRX-Bayes ────────────────┘   under normal video
                                                                                                              (cross-fitted)
                                                         Fisher's method: Σ −log p ──> causal median ──> alarm
```

- **Training:** normal videos only.
- **Calibration:** each channel's frame score is turned into an upper-tail probability against cross-fitted scores of
  normal training frames. The tail is empirical below the 90th percentile and exponential (POT) above it.
- **Fusion:** channels are combined with Fisher's method.
- **Fixed settings:** one set for all datasets (8 px cells, 48 px score window, K = 5, r = 16). The channels were
  chosen leave-one-dataset-out.

## Results (frame-level AUC)

| Protocol | UCSD Ped2 | UCSD Ped1 | CUHK Avenue |
|---|---|---|---|
| **Live**: causal median over 9 frames, nothing taken from the test video | **0.929** | **0.775** | **0.881** |
| Literature: per-video min-max + centred Gaussian (σ = 3) | 0.965 | 0.780 | 0.868 |

- **Seeds:** across three seeds the spread is at most ±0.004.
- **Speed:** the whole pipeline runs on 4 CPU threads at batch size 1 — 21.5 FPS at 180×320 and 12.9 FPS at 240×360.
- **MNAD (CVPR 2020), retrained with its official code:** Ped2 0.940 under its own protocol and 0.842 live; Avenue 0.877
  and 0.897.

### Comparison with a vision-language model (live protocol)

The comparison model is an open, simplified AnomalyRuler-style pipeline (`tools/vlm_score.py`), not AnomalyRuler's
reported numbers:
- Qwen3-VL-8B lists what normal frames show (k = 16).
- Each test frame is described, then judged against that list.

| | Ped2 | Ped1 | Avenue | Time per frame | Hardware |
|---|---|---|---|---|---|
| This method | 0.929 | 0.775 | 0.881 | 0.05–0.08 s | 4-core CPU |
| Qwen3-VL-8B, describe-then-judge | 0.939 | 0.751 | 0.684 | ~1.9 s | RTX 3090 Ti, 18 GB |

The two approaches suit different anomalies:
- The VLM names object-type anomalies such as bicycles and cars.
- It misses motion anomalies such as running and throwing, which the optical-flow channels capture.

## Quick start

```bash
pip install -r requirements.txt
export VAD_CACHE=$PWD/cache
sh scripts/prepare_data.sh        # downloads UCSD Ped1/Ped2 and CUHK Avenue, builds the frame caches
sh scripts/reproduce.sh           # trains the autoencoders, evaluates all channels, prints the protocol table
sh scripts/vlm.sh                 # optional: the vision-language-model comparison (GPU with ≥ 20 GB)
```

## Layout

| Path | Content |
|---|---|
| `aean/` | Models (AEAN 1d/2d/3d), data sampling, scoring (REM, FRX, FRX-Bayes, WRX), channels (optical flow GMM-UBM / RX, deep features) |
| `train.py`, `evaluate.py` | Training on normal video; evaluation with every detector and channel, saving cross-fitted normal reference scores |
| `prepare_*.py` | Dataset caches: UCSD, Avenue, ShanghaiTech, IPAD |
| `tools/fuse_runs.py` | Channel normalisation (z / Fisher) and leave-one-dataset-out channel selection |
| `tools/temporal_protocols.py` | Final score under the live, fixed-lag and literature protocols |
| `tools/vlm_score.py`, `tools/vlm_eval.py` | Vision-language-model baseline and comparison, including a gated cascade |
| `live.py` | Recording a camera, training on it and scoring a live stream (RTSP, file or webcam) |
| `bench.py`, `tools/bench_pipeline.py` | CPU/GPU timing |

## Notes on protocols

Most published results normalise scores per test video and smooth them with a centred window. Both use frames from
the future and statistics of the whole test video, which a live camera does not have. This repository reports both
protocols. The live one is the headline.

## Citation

A paper is in preparation; the citation will be added here.

## License

MIT (see `LICENSE`). The datasets keep their own licenses; download them from the original sources.

## Acknowledgements

MSc thesis work at Gebze Technical University, Department of Electronics Engineering, supervised by
Assoc. Prof. Koray Kayabol.
