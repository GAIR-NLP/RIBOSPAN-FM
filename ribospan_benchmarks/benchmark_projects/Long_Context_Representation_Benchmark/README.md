# Long-Context Representation Benchmark

Cosine geometry, representation diffusion, and attention diagnostics for RNA
language models on paired structured/native transcripts at multiple lengths.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_long_context.py all --device cuda:0
```

Outputs go to `outputs/length_sweep/`. HydraRNA is used in cosine only.

`run_long_context.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `generate`, `attention`, `cosine`, `analyze`, or `all` |
| `--device DEVICE` | Device for attention/cosine |
| `--models NAME ...` | Run a subset of models |
| `--skip-generate` | Keep the existing pair file |
| `--skip-attention` | Skip attention |
| `--skip-analyze` | Skip analysis |
| `--no-resume` | Recompute cosine from scratch |
| `--summary-only` | Attention summaries only, no heatmaps |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
