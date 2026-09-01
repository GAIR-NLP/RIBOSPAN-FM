# RIBOSPAN Benchmarks

```text
ribospan_benchmarks/
  setup.sh
  envs/benchmark.yml
  model_src/
  model_weights/
  benchmark_projects/
    Long_Context_Representation_Benchmark/
    RNA_Type_Representation_Benchmark/
    mRNABench_Linear_Probe_Benchmark/
    RNAGym_Fitness_Benchmark/
```

## Setup

```bash
./setup.sh
```

`--weights-only` skips the conda env; `--env NAME` sets the env name (default `benchmark`). Public checkpoints land in `model_weights/`; place RIBOSPAN under `model_weights/RIBOSPAN-*/`.

`setup.sh` only fetches third-party weights from their official release URLs; their use remains subject to the original providers' licenses and terms.

Do not install packages by hand (`pip`, `conda install`, or `conda env create -f envs/benchmark.yml`) unless you know exactly what you are doing: `mamba-ssm` and `flash-attn` need the flags in `setup.sh`.

## Projects

Reproduce from the project README after `./setup.sh`:

- [Long-Context Representation Benchmark](benchmark_projects/Long_Context_Representation_Benchmark/README.md) — cosine geometry, diffusion, and attention on complete mRNAs
- [RNA Type Representation Benchmark](benchmark_projects/RNA_Type_Representation_Benchmark/README.md) — frozen 10-NN / t-SNE on biotypes and Rfam families
- [mRNABench Linear Probe Benchmark](benchmark_projects/mRNABench_Linear_Probe_Benchmark/README.md) — frozen last-layer mean-pool probes on HL, MRL, eCLIP, GO, VEP, and TE
- [RNAGym Fitness Benchmark](benchmark_projects/RNAGym_Fitness_Benchmark/README.md) — frozen masked-marginal MLM scores on RNAGym DMS assays
