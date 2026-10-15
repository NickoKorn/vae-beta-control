# Fixed-β VARIX evaluation

This project evaluates how fixed β changes e. g. posterior use, information capacity,
representation utility, and posterior collapse in VARIX models for
high-dimensional multi-omics cancer data. AUTOENCODIX is installed as a pinned external dependency; its source is
not included here.

## Start on a new computer

Install **Git and Python 3.11** first. Linux is the currently tested platform.
From a terminal:

```bash
git clone https://github.com/NickoKorn/vae-beta-control.git
cd vae-beta-control
python3.11 scripts/setup.py
.venv/bin/python -m jupyterlab START_HERE.ipynb
```

On Windows, use `py -3.11 scripts/setup.py` and
`.venv\Scripts\python.exe -m jupyterlab START_HERE.ipynb`.
Select **VAE beta control (Python 3.11)** and run the cells in order.
The first installation downloads PyTorch and AUTOENCODIX dependencies and can
be large. Git and network access are needed during setup; no API keys are needed.

The start notebook generates synthetic data and runs a small CPU experiment
using native AUTOENCODIX annealing. Outputs remain local under
`beta_vae_experiments/outputs/quickstart_circle/`. No external disk, TCGA download,
or saved historical results are needed. Increase epochs in a copied JSON for
scientific experiments; three epochs only verify the technical workflow.

`beta_vae_experiments/requirements.txt` is the single main dependency list. Setup installs
GraphRicciCurvature separately with `--no-deps` because its declared SciPy ceiling
conflicts with the tested project stack; its dependencies are listed explicitly.
This compatibility exception applies to the tested Forman path, not a blanket
validation of all package functionality. Local giotto-tda homology is disabled
in the quickstart and is not installed by this setup. Research environments
and older notebooks can require additional dependencies and saved artifacts.
A fresh installation on Windows/macOS has not yet been validated.

A Cookiecutter template is unnecessary for this single repository: cloning and
one setup command provide the entry point without another project generator.

## Installation and execution

Use Python 3.11 and install the pinned dependencies:

```shell
python3.11 scripts/setup.py
```

All commands below are run from the repository root. First verify CUDA in the
same shell that will launch training:

```shell
export CUBLAS_WORKSPACE_CONFIG=:16:8
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The default sequence selects these three profiles from the central experiment catalog; shared defaults and further profiles are in `user_interface/experiments.json`:

- `beta_vae_experiments/user_interface/experiments.json#training.tcga`
- `beta_vae_experiments/user_interface/experiments.json#training.circle`
- `beta_vae_experiments/user_interface/experiments.json#training.swiss_roll`

Each file controls its dataset, run grid, metrics, plots, metric analysis,
outputs, and resume behavior. Validate one experiment without training:

All six metric families use immutable, validated configuration dataclasses.
Optional JSON sections under `metrics` are `graph`, `dimensionality`, `geometry`,
`latent_usage`, `topology`, and `local_topology`; omitted fields retain their
existing defaults. For example, `"geometry": {"selected_k": 20}` sets the
neighborhood size, and `"latent_usage": {"collapse": {"delta": 0.01}}` sets
the posterior-collapse tolerance. Overrides apply to that experiment only.

For TCGA, `training.downstream_targets` selects annotation columns for native
AUTOENCODIX evaluation. The biology configuration selects
`CANCER_TYPE_ACRONYM`, `AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT`, `SEX`, and `AGE`.
These are also the defaults when the field is omitted. Classification scores
are saved as `roc_auc_ovo`; the numeric age task is saved as `r2`. Evaluation
also saves per-class one-vs-rest AUCs in `z_class_auc_use_split.csv` under each
run's downstream artifact directory and prints the five lowest class AUCs per
target and validation/test split. These use the same native fitted classifier;
class counts accompany the scores. Per-class OvR AUC differs from the overall
multiclass OvO AUC and is descriptive, not a model-selection criterion on test data.
Evaluation
still uses sampled `z` and the existing train/validation/test splits. Missing
labels are handled by AUTOENCODIX per task. Synthetic runs skip downstream
evaluation. Existing completed runs are not automatically recomputed; use a
new output directory or explicitly rerun them to obtain the added endpoints.

```shell
./.venv/bin/python -m beta_vae_experiments.new_beta_varix_experiments \
  beta_vae_experiments/user_interface/experiments.json#training.tcga --preflight
```

Run or resume that experiment:

For mean-reduced KL, `training.beta_over_d` can replace `training.betas`:
the runner computes nominal `beta = beta_over_d * latent_dim`. Specify only
one grid; `beta_over_d` requires `losses: ["kl"]` and
`loss_reductions: ["mean"]`. The optional
`user_interface/experiments.json#training.tcga_beta_over_d` uses the historical
beta grid divided by 8, preserving nominal beta at latent dimension 8, and
writes to a separate result directory. Sample-ID split loading is not yet
integrated into the runner. TCGA launches require `PYTHONHASHSEED=2`;
each output directory saves its actual split IDs and rejects a changed split on resume.

Interface configurations use an explicit `results_date` (YYYY-MM-DD) to write
to `beta_vae_experiments/new_results_<date>/<experiment_name>/`. Keep the date
unchanged when resuming; change it for a new dated experiment. Historical
configuration snapshots without `results_date` retain their original output path.
The directory contains `plots/`, `metric_analysis/`, `metrics_artifacts/`,
`posterior_snapshots/` and `run_analysis/`. The latter stores `training_grid.csv`
and, per run, `epoch_losses.csv` plus `posteriors/posterior_train.npz`,
`posterior_valid.npz` and `posterior_test.npz` (mu, logvar, sample_ids).
All three exports use fresh inference with the same final model in eval mode.
Epoch histories retain native AUTOENCODIX losses and additionally record
sample-averaged reconstruction MSE and unweighted total KL, including at beta=0.
These exports enable offline downstream evaluation; they do not change the
existing sampled-z downstream endpoint.

```shell
./.venv/bin/python -m beta_vae_experiments.new_beta_varix_experiments \
  beta_vae_experiments/user_interface/experiments.json#training.tcga
```

Run biology, Circle, and Swiss Roll sequentially:

```shell
./beta_vae_experiments/bash_scripts/run_biology_circle_swiss_roll.sh
```

The script checks CUDA and validates all three JSON files before training. It
stops if one experiment fails; completed runs remain resumable according to
the corresponding JSON configuration.

## Metric analysis and plots

`new_beta_varix_experiments.py` performs the configured run-level metrics and
the combined metric analysis after training. The existing standalone analysis
command remains available for historical result folders:

```shell
./.venv/bin/python -m beta_vae_experiments.result_analysis.sweep_analysis \
  --input v5_core_transition_main_with_plots/fixed_beta_run_metrics_absolute_core_v4_cuda.csv \
  --output-dir v5_core_transition_main_with_plots/analysis_v5
```

This creates rate-distortion and collapse summary tables plus:

- rate-distortion trajectories with Pareto points;
- posterior-SNR trajectories;
- effective-dimension trajectories;
- LCMC geometry trajectories;
- RTD and normalized H1-Wasserstein topology trajectories;
- posterior-collapse trajectories.

These are run-spanning Core plots. AUTOENCODIX sampled-z plots, posterior-mean
PCA/2D views, and test-split ridgelines are additionally generated per run.

The existing results table is indexed by the complete training identity.
Configurations recorded with status `ok` are skipped; failed and new tuples are
trained again.

The AUTOENCODIX commit in `requirements.txt` is part of the scientific
provenance. Any upgrade requires a new loss audit and complete local test run.
Raw data, patient-level artifacts, checkpoints, generated results, and caches
are excluded from version control.

## Active repository scope

The active code consists of `metrics/`, `datasets/`, `user_interface/`,
`data_pipeline.py`, `metric_pipeline.py`, `new_beta_varix_experiments.py`, and
`run_artifacts.py`. `run_artifacts.py` is required by training for epoch histories,
final posterior exports, artifact directories, and split consistency checks.
`bash_scripts/run_biology_circle_swiss_roll.sh` is the current launch script;
it also accepts experiment JSON paths as arguments.

`result_analysis/sweep_analysis.py` is called by the training runner.
`analysis.ipynb` contains presentation plots inline, and
`additional_downstream_tasks.ipynb` contains additional task analysis.
Both notebooks are versioned without executed outputs; executed copies remain local.
Historical analysis notebooks and one-off scripts are retained in local archives.
See `result_analysis/README.md` for active analysis entry points.

MRRE (false/missing) is executed by the active ZADU geometry pipeline.
Directed edge overlap and the retired posterior-mean/PCA plot implementation
remain in local historical archives, not in the active metric pipeline.
Historical result columns remain intact; partition NMI/ARI continue to be active
graph comparisons. The standard workflow keeps AUTOENCODIX's sampled-z plots
and class-mean stars.

Historical scripts, complementary metrics, controller experiments, and their
tests are preserved locally in `archiv_06_10/`; the previously created
`archive_06_10_26/` also remains local. Both archives are excluded from Git,
along with generated results, plots, checkpoints, and local datasets.

Run the active regression tests with:

```shell
./.venv/bin/python -m pytest beta_vae_experiments/test -q
```
