# Saved-experiment analysis

These tools evaluate saved artifacts; the annealing preparer only writes configurations.

| Module | Purpose |
| --- | --- |
| `sweep_analysis.py` | Training-integrated rate–distortion and collapse summaries |
| `analyse_annealing_snapshots.py` | Rate, preservation and task evaluation across epoch snapshots |
| `prepare_annealing_recovery.py` | Generate JSON configurations and beta schedule previews |
| `encoder_geometry_analysis.py` | Evaluate saved encoder/posterior geometry |
| `validate_encoder_geometry.py` | Validate smooth curvature formulas on known synthetic surfaces |
| `spectral_dimension_analysis.py` | Spectral profiles, ablation and held-out trajectory analysis |
| `spectral_dimension_followup.py` | Additional spectral stability and forecasting analysis |
| `utils.py` | Shared posterior reader, file digest and descriptive statistics |

Data download: `datasets/fetch_tcga_pancancer.py`.
External result copying: `scripts/publish_annealing_results.py`.
Main notebooks: `../analysis.ipynb` and `../additional_downstream_tasks.ipynb`.
Historical analyses are retained in the ignored local archive, not required by training.
