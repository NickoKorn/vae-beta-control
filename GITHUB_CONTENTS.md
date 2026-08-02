# GitHub content policy

The public project is a separate fixed-beta experiment repository.
AUTOENCODIX remains an external dependency and its source tree is not copied.

## Include

- `beta_vae_experiments/README.md`
- `beta_vae_experiments/__init__.py`
- the experiment runner, data pipeline, resumable trainer, and references module
- core Python modules under `beta_vae_experiments/metrics/`
- reproducibility shell scripts
- `beta_vae_experiments/requirements.txt`

## Exclude for the initial publication

- the parent repository's `src/autoencodix/`, tests, tutorials, documentation,
  and configurations
- `beta_vae_experiments/tests/`
- `beta_vae_experiments/optional/`
- research notes and metric documentation stored as `.txt`
- raw or processed data, patient-level artifacts, outputs, plots, checkpoints,
  logs, caches, notebooks, editor settings, and local environments

`requirements.txt` is the only intended `.txt` exception because it pins the
external AUTOENCODIX implementation and evaluation dependencies.

## Verification

Before exporting or committing the allowlisted files, run locally:

```bash
git status --short
git ls-files --others --exclude-standard
git diff --cached --name-status
git diff --cached --stat
```

The complete local test and lint suites must pass before publication even
though their source files are not included in the initial public repository.
No sensitive or patient-level result may be committed.
