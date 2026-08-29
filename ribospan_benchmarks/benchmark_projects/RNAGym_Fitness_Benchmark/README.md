# RNAGym Fitness Benchmark

Frozen masked-marginal MLM scores on the RNAGym DMS assays
(growth, cleavage, binding, splicing, …). No linear probe, no fine-tuning.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_fitness.py all --device cuda:0
```

To reproduce these numbers, use the frozen tables already under `data/`
(official RNAGym processed assays and `reference_sheet_final.csv` DMS IDs;
substitutions only, U→T, WT from the reference sheet). Please cite Arora et al.
2025, https://doi.org/10.1101/2025.06.16.660049. Per-assay scores go to
`outputs/rnagym_fitness/results/`; summaries go to `outputs/rnagym_fitness/`.

`run_fitness.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `score`, `summarize`, or `all` |
| `--group NAME` | `all` (default), `smoke`, `ribozyme`, `trna`, `aptamer`, `mrna-coding`, `mrna-splicing` |
| `--datasets ID ...` | Official `DMS_ID` subset |
| `--models NAME ...` | Registry models |
| `--device DEVICE` | Score device |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
