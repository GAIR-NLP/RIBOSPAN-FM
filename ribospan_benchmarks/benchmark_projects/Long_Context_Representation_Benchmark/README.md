# Long-Context Representation Benchmark

Cosine geometry, representation diffusion, masked-nucleotide accuracy, and
attention diagnostics for RNA language models on paired structured/native
transcripts at multiple lengths.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_long_context.py all --device cuda:0
```

Outputs go to `outputs/length_sweep/`. HydraRNA is not used in attention.

`mlm` masks 15% of each native transcript and scores how often the top
nucleotide is the original one. `restricted_attention` repeats the cosine
endpoints and that accuracy with every attention layer held to the model's
pretraining span, the `context_window` in `configs/models.yaml`. HydraRNA has
attention in two of its twelve layers; only those are held. The models under
`restricted_attention.span_sweep` in `configs/experiment.yaml` are also scored
at each listed shorter span; `analysis/attention_span_sweep_scores.csv` shows
how their accuracy changes as the span narrows.

`run_long_context.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `generate`, `attention`, `cosine`, `mlm`, `restricted_attention`, `analyze`, or `all` |
| `--device DEVICE` | Device for the model stages |
| `--models NAME ...` | Run a subset of models |
| `--skip-generate` | Keep the existing pair file |
| `--skip-attention` | Skip attention |
| `--skip-mlm` | Skip mlm |
| `--skip-restricted-attention` | Skip restricted_attention |
| `--skip-analyze` | Skip analysis |
| `--no-resume` | Recompute cosine, mlm, and restricted_attention from scratch |
| `--summary-only` | Attention summaries only, no heatmaps |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
