# mRNABench Linear Probe Benchmark

Frozen last-layer mean-pool embeddings with RidgeCV / logistic probes on
mRNABench sequence tables (HL, MRL, eCLIP, GO, VEP) plus translation
efficiency.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_mrnabench.py all --device cuda:0
```

To reproduce these numbers, use the frozen tables already under `data/`
(HuggingFace `morrislab` via `mrna-bench==1.2.2` and the catalogue IDs in
`configs/experiment.yaml`; TE from `morrislab/translation-efficiency-*`;
splits follow Shi et al. as closely as practical). Please cite Shi et al.
2025, https://doi.org/10.1101/2025.07.05.662870. Embeddings go to
`outputs/mrnabench/embeddings/`; probe metrics go to
`outputs/mrnabench/results/`.

`run_mrnabench.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `embed`, `probe`, `summarize`, or `all` |
| `--models NAME ...` | Subset of registry models |
| `--device DEVICE` | Embed device |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
