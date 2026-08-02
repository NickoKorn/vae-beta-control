# Fixed-β VARIX evaluation

This project evaluates how fixed β changes posterior use, information capacity,
representation utility, and posterior collapse in VARIX models for
high-dimensional multi-omics cancer data. It follows a
preregistration-compatible design and does not construct a global quality
score. AUTOENCODIX is installed as a pinned external dependency; its source is
not included here.

## Research questions

1. How does β alter posterior usage, rate, mutual information, total
   correlation, and effective dimensionality?
2. What is the largest tested β preserving every prespecified primary utility
   endpoint within its practical non-degradation margin?
3. What compression is gained inside that safe region?
4. Across which tested β interval does formal posterior collapse emerge?
5. Are observed distortion–rate trajectories broadly consistent with
   minimizing `D + βR`?

Input-space preservation is described as such and is not interpreted as
biological validity.

## Experimental design

β is varied while architecture, preprocessing, split identity, latent
dimension, and other settings remain fixed inside each comparison stratum. A
stratum contains compatible runs with the same loss family, loss reduction,
learning rate, and latent dimensionality. Incompatible strata are never merged
into one β-response curve.

The initial design uses one training seed. Therefore individual values and
reference differences are available, but seed-level confidence intervals and
claims of reproducibility are not. Collapse brackets are labelled
`single_seed_provisional`, and `beta_max_non_degraded` remains uncertified.

## Split locking

```text
train       model fitting
validation  β selection and all analysis choices
test        final evaluation after choices are frozen
```

Test data must not choose β, margins, thresholds, primary metrics,
neighborhood size, knees, or smoothness definitions. If required validation
metrics or preregistered margins are unavailable, selection remains locked.

## Audited VARIX objective

For KL-trained VARIX, the implemented batch objective is

```text
L = reconstruction_loss + β × KL_loss
KL = 0.5 × (μ² + exp(logvar) - logvar - 1).
```

Relative to conventional per-sample feature-summed distortion `D` and
latent-summed rate `R`:

```text
sum reduction:   β_effective = β_nominal
mean reduction:  β_effective = β_nominal × input_dimension / latent_dimension
```

With `mean`, MSE averages over samples and features while KL averages over
samples and latent dimensions. β=1 is canonical only when the resulting
relative scaling matches the standard ELBO convention. MMD runs are a separate
objective family and receive no KL-based effective-β interpretation.

## Metric hierarchy

The primary endpoints remain separate and must all pass:

- one preregistered reconstruction error;
- Trustworthiness between preprocessed input `X` and posterior mean `μ`;
- Continuity between preprocessed input `X` and posterior mean `μ`.

Secondary diagnostics include rate, index-code mutual information, β-TCVAE
total correlation, dimension-wise aggregated KL, per-dimension conditional KL,
ε-δ collapse, Active Units, effective rank, the full centered-μ PCA spectrum,
intrinsic dimension, and sampled-`z` geometry. Secondary metrics describe
compression and failure modes but cannot veto the primary utility decision.

## Posterior collapse

For latent dimension `j`:

```text
KL_j(x) = 0.5 × [μ_j(x)² + exp(logvar_j(x)) - logvar_j(x) - 1].
```

It is ε-δ collapsed when

```text
P_x(KL_j(x) < ε) >= 1 - δ.
```

Collapse is defined from this posterior diagnostic—not from reconstruction,
Active Units, mutual information, or rank alone. Each compatible stratum
reports the last tested β before strong collapse, the tested transition
bracket, and the first tested β meeting the strong-collapse criterion. With one
seed, this is an observed bracket rather than a reproducible population
threshold.

## Information decomposition

Chen, Li, Grosse, and Duvenaud (2018) derive

```text
R = I(X;Z) + TC(Z) + Σ_j KL(q(z_j) || p(z_j)).
```

The pipeline reports the numerical sanity check

```text
decomposition_residual = R - (MI + TC + DWKL).
```

The residual is project-derived from the theoretical identity; it is not a
separate metric proposed by Chen et al. Rate and mixture-density components use
the same deterministic subset. Finite-sample and Monte Carlo estimation mean
that an exactly zero residual is not expected.

Reference: Chen, R. T. Q., Li, X., Grosse, R. B., and Duvenaud, D. K. (2018),
“Isolating Sources of Disentanglement in Variational Autoencoders,” NeurIPS 31.

## Distortion–rate diagnostic

For ideal global minimizers and `β₁ < β₂`:

```text
R₂ <= R₁
D₂ >= D₁
β₁(R₁ - R₂) <= D₂ - D₁ <= β₂(R₁ - R₂).
```

When `R₁ > R₂`:

```text
secant_beta = (D₂ - D₁) / (R₁ - R₂).
```

The diagnostic uses objective-compatible D, R, and effective β. Violations are
reported unchanged as ideal-global-optimum consistency diagnostics, not as
automatic model failures or estimates of a true optimal β.

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
