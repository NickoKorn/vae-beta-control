"""Downstream predictive evaluations for trained beta-VAE representations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier

from autoencodix.varix import Varix


def _save_evaluation_table(frame: pd.DataFrame, dir: Path) -> None:
    """Persist one downstream evaluation table in portable tabular formats."""
    frame.to_csv(dir.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(dir.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError):
        pass


def save_random_forest_cv_evaluation(
    varix: Varix,
    output_dir: Path,
    run_name: str,
    parameter: str = "CANCER_TYPE_ACRONYM",
) -> pd.DataFrame:
    """Run and save Random-Forest CV-5 accuracy for latent and random features.

    The evaluation delegates to the trained VARIX pipeline's public
    ``evaluate`` method. It compares the sampled latent representation against
    AUTOENCODIX random-feature references for one downstream classification
    target and saves only the rows added by this call.
    """
    downstream_dir = Path(output_dir) / "downstream" / run_name
    downstream_dir.mkdir(parents=True, exist_ok=True)

    existing = getattr(varix.result, "embedding_evaluation", pd.DataFrame())
    previous_rows = len(existing) if isinstance(existing, pd.DataFrame) else 0
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=int(varix.config.global_seed),
    )
    # AUTOENCODIX requests joblib n_jobs=-1 internally. The threading backend
    # avoids process/semaphore creation and duplicating the trained pipeline in
    # child processes while preserving the evaluate() cross-validation path.
    with parallel_backend("threading"):
        evaluation_result = varix.evaluate(
            params=[parameter],
            ml_model_class=classifier,
            metric_class="accuracy",
            split_type="CV-5",
            reference_methods=["RandomFeature"],
        )
    combined = getattr(evaluation_result, "embedding_evaluation", None)
    if not isinstance(combined, pd.DataFrame) or len(combined) <= previous_rows:
        raise ValueError("Random-Forest CV evaluation returned no new result rows.")

    evaluation = combined.iloc[previous_rows:].reset_index(drop=True)
    _save_evaluation_table(
        evaluation,
        downstream_dir / "random_forest_cv5_accuracy_evaluation",
    )
    config_path = downstream_dir / "random_forest_cv5_accuracy_config.json"
    config_path.write_text(
        json.dumps(
            {
                "parameter": parameter,
                "classifier": "RandomForestClassifier",
                "n_estimators": 100,
                "random_state": int(varix.config.global_seed),
                "metric_class": "accuracy",
                "split_type": "CV-5",
                "reference_methods": ["RandomFeature"],
                "representation": "sampled_z_from_result_latentspaces",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return evaluation
