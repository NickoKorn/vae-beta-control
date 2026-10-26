# VAE beta control

JSON-configured AUTOENCODIX/VARIX experiments for studying how beta compression
changes reconstruction, latent usage, geometry, graphs, topology and downstream
performance. Start with the small CPU example before preparing larger experiments.

## Requirements and tested environment

- Python **3.11**, Git, an internet connection and a 64-bit machine.
- Tested locally on Linux with PyTorch **2.3.1+cu121**, NumPy **2.3.5**, and an
  NVIDIA RTX 4070. The quickstart uses the CPU and needs no downloaded dataset.
- GPU training needs a compatible NVIDIA driver and PyTorch build. A separate
  CUDA Toolkit is not required for the prebuilt PyTorch packages.
- The pinned AUTOENCODIX package requires `torch>=2.1.0,<2.4.0` and
  `numpy>=2.0.2`. Do not independently upgrade PyTorch beyond that bound.

The setup script installs the main requirements and then the graph overlay with
`--no-deps`. This is deliberate: GraphRicciCurvature declares an older SciPy
ceiling, while our active Forman path has been tested with the project's newer
SciPy. Use the setup script rather than installing both requirement files in one
unrestricted resolver invocation.

## Linux installation

Run commands from a terminal. On Debian/Ubuntu, install these prerequisites if
needed:

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake
```

Clone the repository:

```bash
git clone https://github.com/NickoKorn/vae-beta-control.git
cd vae-beta-control
```

If Python 3.11 is already installed:

```bash
python3.11 scripts/setup.py
```

If Python 3.11 is missing, [uv](https://docs.astral.sh/uv/getting-started/installation/)
can create the required environment without replacing your system Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv --python 3.11 --seed .venv
.venv/bin/python scripts/setup.py
```

The setup creates `.venv`, installs dependencies, registers a Jupyter kernel and
checks the experiment-runner imports. It does not download TCGA or start training.

## Windows 11: native PowerShell or WSL2

### Native PowerShell setup

Install 64-bit [Python 3.11](https://www.python.org/downloads/) and
[Git for Windows](https://git-scm.com/downloads/win), then open a new PowerShell
terminal:

```powershell
git clone https://github.com/NickoKorn/vae-beta-control.git
cd vae-beta-control
py -3.11 scripts/setup.py
```

The setup script supports Windows executable paths. Activation is unnecessary;
use the `.venv` executable paths shown below.
Before training, check that NumPy and PyTorch can exchange arrays:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np, torch; a=np.ones((2, 2), dtype=np.float32); print(torch.from_numpy(a).numpy())"
```

**Native Windows is not yet verified end to end.** PyTorch 2.3.1 Windows builds
have a documented NumPy 2 interoperability issue (`_ARRAY_API not found` or
`Numpy is not available`). Our current dependency bounds include that combination.
An import succeeding alone does not establish that training works. If the check
above fails, the current pinned environment needs a separately tested Windows
compatibility change; WSL2 is the alternative for using the Linux instructions.
See the [PyTorch report](https://discuss.pytorch.org/t/getting-runtimeerror-numpy-is-not-available-whenever-trying-with-torch-from-numpy/206871).

If setup and the interoperability check pass, run the CPU example:

```powershell
$env:PYTHONHASHSEED = "2"
.\.venv\Scripts\python.exe -m beta_vae_experiments.new_beta_varix_experiments beta_vae_experiments/user_interface/quickstart.json --preflight
.\.venv\Scripts\python.exe -m beta_vae_experiments.new_beta_varix_experiments beta_vae_experiments/user_interface/quickstart.json
```

To open the notebook interface:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab START_HERE.ipynb
```

The Bash suite launcher is for Linux/WSL2. Under native Windows, start individual
JSON files with the Python module commands above.

### WSL2 alternative

In PowerShell **as administrator**:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested, launch Ubuntu, create its user account, and follow
the **Linux installation** instructions inside Ubuntu. Clone into your Linux home
folder. This uses Linux packages on Windows; WSL2 itself has not been tested in
this repository's local verification runs.
[Microsoft installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

## Start the CPU example on Linux/WSL2

All commands below run from the cloned repository root:

```bash
export PYTHONHASHSEED=2
.venv/bin/python -m beta_vae_experiments.new_beta_varix_experiments beta_vae_experiments/user_interface/quickstart.json --preflight
.venv/bin/python -m beta_vae_experiments.new_beta_varix_experiments beta_vae_experiments/user_interface/quickstart.json
```

`--preflight` validates configuration and paths without training. The second
command trains one circle VAE for three epochs and computes its metrics. Results
are written to `beta_vae_experiments/outputs/quickstart_circle_3d/`.

For notebooks:

```bash
.venv/bin/python -m jupyterlab START_HERE.ipynb
```

Choose **VAE beta control (Python 3.11)** and execute the cells in order.
`analysis.ipynb` and `additional_downstream_tasks.ipynb` require existing research
results; use `START_HERE.ipynb` for a fresh clone. Those analysis notebooks expect
the repository root as their working directory.

## Configure an experiment and export epoch snapshots

Copy `beta_vae_experiments/user_interface/quickstart.json`, choose a new
`experiment_name` and `output_dir`, and edit the training settings. Change
`training.device` from `cpu` to `cuda` only after checking GPU availability.
Set `PYTHONHASHSEED=2` before starting Python; TCGA explicitly requires it.

To save native Train/Validation arrays every 50 epochs, add this inside `training`:

```json
"snapshots": {
  "enabled": true,
  "every": 50,
  "downstream": true
}
```

This sets the native checkpoint interval and keeps training dynamics in RAM.
After training, the runner exports recorded `mu`, `logvar`, `z` and sample IDs
and optionally evaluates their downstream tasks. It also exports the final
epoch if it is not a multiple of the interval. There is no custom trainer or
additional encoder inference for epoch export. Native Train arrays were captured
during batch updates; Validation arrays during validation. Missing native arrays
are recorded in `snapshot_index.csv`, not replaced with synthetic zeros.

CLI options override JSON settings:

```bash
.venv/bin/python -m beta_vae_experiments.new_beta_varix_experiments path/to/experiment.json --epoch-snapshots --snapshot-every 50
```

Use `--no-epoch-snapshots` to disable epoch export or
`--no-snapshot-downstream` to export without epoch downstream evaluation.

## Download TCGA

Circle, Sphere and Swiss-Roll variants are generated locally. TCGA is separate
from the Git repository. Download the three required Parquet files from the
official [AUTOENCODIX Zenodo archive](https://zenodo.org/records/15518831):

```bash
.venv/bin/python beta_vae_experiments/datasets/fetch_tcga_pancancer.py beta_vae_experiments/datasets/manifests/tcga_pancancer_source.json --destination data/raw/TCGAPanCancerAutoencodix
```

For PowerShell, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.
The downloader resumes partial transfers and verifies file size and archive CRC.
Allow roughly 5 GB of free space for these files and retained download data;
training outputs need additional space. Set the JSON dataset option `data_root`
to this directory. Do not copy absolute paths from another machine.

## Prepare and run the large annealing grid

This is separate from the small quickstart. The current suite prepares 18 JSON
files for 384 runs, each with 1000 epochs and a 300-epoch linear beta warm-up.
Preparing files does not start training:

```bash
.venv/bin/python -m beta_vae_experiments.prepare_json_for_training --output-dir beta_vae_experiments/large_scale_results
```

Choose a fresh output directory when preparing another grid; the generator
refuses to overwrite existing JSON configurations. Its suite definition is
`beta_vae_experiments/user_interface/experiments.json#annealing.warmup`.

On Linux/WSL2, check GPU availability, fetch TCGA, then use the suite launcher:

```bash
bash beta_vae_experiments/bash_scripts/run_annealing.sh --preflight beta_vae_experiments/large_scale_results/configs
bash beta_vae_experiments/bash_scripts/run_annealing.sh --run beta_vae_experiments/large_scale_results/configs
```

The launcher requires CUDA and validates every configuration before training.
For native PowerShell, prepare the grid using the Windows Python path, then run
its JSON files individually; the Bash launcher is not required.

## Results and checks

Each experiment writes `experiment_config.json`, `split.json` and
`experiment_metrics.csv`. Main per-run outputs include:

- `run_analysis/<run>/epoch_losses.csv` and `model_final.pt`.
- `posterior_snapshots/<run>/posterior_{train,valid,test}.npz` with `mu`, `logvar`
  and `sample_ids` from final-model inference.
- `run_analysis/<run>/epoch_snapshots/` when epoch export is enabled.
- `absolute_core_metrics_v1/` and `metrics_artifacts/` with metric tables and
  diagnostic artifacts. Undefined measures have explicit status/warning fields.

Final downstream evaluation includes one stored z representation and additional
posterior draws summarized by the **median of their scores**. Epoch evaluations
use Train/Validation; the final evaluation can use Test. Completed successful runs
are skipped when `execution.resume_existing` is true and `rerun_passed` is false.
Changing saved split IDs/order is rejected before resuming.

Check the tensor/NumPy bridge and CUDA build on Linux/WSL2:

```bash
.venv/bin/python -c "import numpy as np, torch; print(torch.from_numpy(np.ones((2, 2), dtype=np.float32)).numpy()); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The locally tested GPU build is PyTorch 2.3.1 with CUDA 12.1. If appropriate for
your GPU, it can be selected **before** running setup:

```bash
.venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Use the Windows Python path in PowerShell. Other GPU generations may need a
newer PyTorch build and therefore a coordinated change to the AUTOENCODIX pin;
installing a newer standalone CUDA Toolkit does not change the PyTorch build.
[Official PyTorch version/build matrix](https://docs.pytorch.org/get-started/previous-versions/).

Experiments, downloaded data, caches and local `metrics/complementary/` methods
are intentionally excluded from Git.

### Current large-scale metric scope

The `preservation_only` setting selects Trustworthiness, Continuity, Steadiness,
Cohesiveness and scale-normalized Stress, plus rate and collapse diagnostics.
Graph descriptors, topology, total correlation and dimensionality are deferred.
Neighbor mixing is a separate graph diagnostic with 50 posterior draws. Its
current cancer/sex grouping uses clinical labels; it is not a label-free controller.
Downstream probes use z only, including epoch snapshots. Final repeated-z AUC
uses the median of 10 evaluations. Snapshots are saved every 50 epochs.
