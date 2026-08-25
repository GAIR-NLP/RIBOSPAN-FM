# RNA Type Representation Benchmark

kNN and t-SNE evaluation of frozen RNA language-model embeddings on RNA
biotypes and Rfam families.

Environment and checkpoints: [`../../README.md`](../../README.md).

```bash
conda activate benchmark
python run_rna_type.py all --device cuda:0
```

Outputs go to `outputs/type_separability/`.

`run_rna_type.py`:

| Option | Meaning |
|---|---|
| `STAGE` | `embed`, `rfam`, `analyze`, `plot`, or `all` |
| `--device DEVICE` | Device for embedding |
| `--models NAME ...` | Run a subset of models |
| `--recompute-tsne` | Recompute t-SNE instead of reusing saved coordinates |

Experiment settings: `configs/experiment.yaml`. Model list: `configs/models.yaml`.
