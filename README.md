# Fixed-β VARIX evaluation

This project evaluates how fixed β changes e. g. posterior use, information capacity,
representation utility, and posterior collapse in VARIX models for
high-dimensional multi-omics cancer data. AUTOENCODIX is installed as a pinned external dependency; its source is
not included here.

## Installation and execution

Use Python 3.11 and install the pinned dependencies:

```shell
python -m pip install -r beta_vae_experiments/requirements.txt
```

Run from the repository root:

```shell
export CUBLAS_WORKSPACE_CONFIG=:16:8
python beta_vae_experiments/beta_varix_experiments.py \
  --skip-smoke-test \
  --epochs 50 \
  --loss-reduction sum \
  --device cuda \
  --yes
```

Training requires a CUDA-enabled PyTorch build and an available NVIDIA GPU.
The launcher `beta_vae_experiments/run_full_sum_cuda.sh` sets the deterministic
cuBLAS workspace configuration before Python starts.

By default, the existing results table is indexed by the complete six-value
training tuple (loss, learning rate, beta, seed, epochs, and reduction).
Configurations recorded with status `ok` are skipped; failed and new tuples are
trained. Pass `--rerun-passed` to train successful tuples again.

The AUTOENCODIX commit in `requirements.txt` is part of the scientific
provenance. Any upgrade requires a new loss audit and complete local test run.
Raw data, patient-level artifacts, checkpoints, generated results, and caches
are excluded from version control.
