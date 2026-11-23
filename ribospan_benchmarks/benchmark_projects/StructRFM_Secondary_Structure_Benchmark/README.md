# StructRFM Secondary Structure Benchmark

Frozen-encoder MixedFold training with Zuker decoding on bpRNA and
RNAStrAlign. Published structRFM checkpoints are jointly fine-tuned and used
as the comparison.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_structrfm.py --device cuda:0
```

To reproduce these numbers, use the structRFM v0.0.8 task archive.
`../../setup.sh` downloads it from the structRFM release into
`data/structrfm_release/` and extracts it when the splits are missing. Please
cite Zhu et al. 2026, https://doi.org/10.1038/s41586-026-11210-2. RIBOSPAN heads go to
`outputs/ssp_structrfm/heads/`; published-checkpoint tables go to
`outputs/ssp_structrfm/official/structRFM/<task>/`. A finished
`test_metric_*.csv` is skipped. `summarize` collects the tables into
`outputs/ssp_structrfm/summary.tsv`, with the mean F1, INF (the geometric
mean of precision and recall), precision, and recall. The published structRFM
checkpoint is scored on every ArchiveII sequence and the RIBOSPAN heads on
those up to 599 nt, so the `shared_*` columns give each model's means over the
sequences all models have scored. `secondary_structure_inf.pdf` shows the
per-sequence INF on those shared sequences for the models under
`figure.models` in `configs/experiment.yaml`: a violin per test set with the
median as a line, the mean as a dot and as the number above, and a box per
ArchiveII family.

`run_structrfm.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `check`, `embed`, `train`, `score`, `summarize`, or `all` |
| `--model NAME` | `structRFM` or one RIBOSPAN checkpoint. Default: all. |
| `--task NAME` | `bpRNA1m` or `RNAStrAlign` |
| `--device DEVICE` | Torch device |
| `--epochs N` | Training epochs. Default 60. |
| `--epoch N` | Checkpoint epoch for `score` |
| `--limit N` | Use the first N sequences |
| `--no-skip-existing` | Run even when `test_metric_*.csv` exists |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
