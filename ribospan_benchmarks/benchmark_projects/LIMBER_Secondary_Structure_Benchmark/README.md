# LIMBER Secondary Structure Benchmark

RiNALMo gradual-unfreeze fine-tuning on bpRNA TS0, Rivas TestSetB, and
ArchiveII leave-one-family-out. Sequences longer than 512 nt are dropped.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_limber.py --device cuda:0
```

To reproduce these numbers, use the frozen splits under `data/`. Please cite
Wang et al. 2026, https://doi.org/10.64898/2026.10.28.762670. `best.pt` is the
highest validation flexible F1; test numbers are written once to `result.json`.
A finished `result.json` is skipped. `summarize` collects them into
`outputs/limber_ss/summary.tsv`: mean and median F1 on TS0 and TestSetB, and,
as LIMBER reports ArchiveII, the mean F1 pooled over the test sequences of all
nine held-out families. `summary_archiveII_by_family.tsv` lists each family.
`secondary_structure_flexible_f1.pdf` shows the per-sequence F1 of the models
under `figure.models` in `configs/experiment.yaml`: a violin per test set with
the median as a line and as the number above, the mean as a dot, and a box per
ArchiveII family.

`run_limber.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `check`, `train`, `summarize`, or `all` |
| `--model NAME` | One registry model. Default: all. |
| `--task NAME` | `bprna_ts0`, `rivas_testsetb`, or `archiveII_lfo` |
| `--family NAME` | One ArchiveII family. Default: all nine. |
| `--device DEVICE` | Torch device |
| `--max-train N` | Cap the training split. `0` uses all of it. |
| `--no-skip-existing` | Retrain even when `result.json` exists |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
