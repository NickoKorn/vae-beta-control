"""Scientific references, research questions, and provenance serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from beta_vae_experiments.metrics.registry import REGISTRY
except ModuleNotFoundError:
    from metrics.registry import REGISTRY  # type: ignore[no-redef]


SCIENTIFIC_REFERENCES: tuple[dict[str, str], ...] = (
    {
        "citation_key": "KingmaWelling2014VAE",
        "title": "Auto-Encoding Variational Bayes",
        "authors": "Diederik P. Kingma; Max Welling",
        "year": "2014",
        "identifier": "arXiv:1312.6114",
        "url": "https://arxiv.org/abs/1312.6114",
        "role": "VAE objective, reconstruction, and conditional Gaussian KL.",
    },
    {
        "citation_key": "Alemi2018BrokenELBO",
        "title": "Fixing a Broken ELBO",
        "authors": "Alexander A. Alemi; Ian Fischer; Joshua V. Dillon; Kevin Murphy",
        "year": "2018",
        "identifier": "arXiv:1711.00464; ICML 2018",
        "url": "https://arxiv.org/abs/1711.00464",
        "role": "Rate-distortion interpretation of variational latent-variable models.",
    },
    {
        "citation_key": "Bac2021ScikitDimension",
        "title": "Scikit-Dimension: A Python Package for Intrinsic Dimension Estimation",
        "authors": (
            "Jonathan Bac; Evgeny M. Mirkes; Alexander N. Gorban; "
            "Ivan Tyukin; Andrei Zinovyev"
        ),
        "year": "2021",
        "identifier": "arXiv:2109.02596",
        "url": "https://arxiv.org/abs/2109.02596",
        "role": "Software provenance for TwoNN and MLE intrinsic-dimension estimates.",
    },
    {
        "citation_key": "Facco2017TwoNN",
        "title": "Estimating the intrinsic dimension of datasets by a minimal neighborhood information",
        "authors": "Elena Facco; Maria d'Errico; Alex Rodriguez; Alessandro Laio",
        "year": "2017",
        "identifier": "DOI:10.1038/s41598-017-11873-y",
        "url": "https://doi.org/10.1038/s41598-017-11873-y",
        "role": "TwoNN intrinsic-dimension estimator.",
    },
    {
        "citation_key": "Levina2004MLEID",
        "title": "Maximum Likelihood Estimation of Intrinsic Dimension",
        "authors": "Elizaveta Levina; Peter J. Bickel",
        "year": "2004",
        "identifier": "NeurIPS 17",
        "url": "https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html",
        "role": "Maximum-likelihood intrinsic-dimension estimator.",
    },
    {
        "citation_key": "JolliffeCadima2016PCA",
        "title": "Principal component analysis: a review and recent developments",
        "authors": "Ian T. Jolliffe; Jorge Cadima",
        "year": "2016",
        "identifier": "DOI:10.1098/rsta.2015.0202",
        "url": "https://doi.org/10.1098/rsta.2015.0202",
        "role": "Interpretation of the posterior-mean PCA singular-value spectrum.",
    },
    {
        "citation_key": "Burda2016IWAE",
        "title": "Importance Weighted Autoencoders",
        "authors": "Yuri Burda; Roger Grosse; Ruslan Salakhutdinov",
        "year": "2016",
        "identifier": "arXiv:1509.00519; ICLR 2016",
        "url": "https://arxiv.org/abs/1509.00519",
        "role": "Variance-of-posterior-means active-unit diagnostic.",
    },
    {
        "citation_key": "Lucas2019PosteriorCollapse",
        "title": "Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse",
        "authors": "James Lucas; George Tucker; Roger Grosse; Mohammad Norouzi",
        "year": "2019",
        "identifier": "arXiv:1911.02469",
        "url": "https://arxiv.org/abs/1911.02469",
        "role": "Epsilon-delta posterior-collapse definition.",
    },
    {
        "citation_key": "Chen2018BetaTCVAE",
        "title": "Isolating Sources of Disentanglement in Variational Autoencoders",
        "authors": "Ricky T. Q. Chen; Xuechen Li; Roger Grosse; David Duvenaud",
        "year": "2018",
        "identifier": "arXiv:1802.04942",
        "url": "https://arxiv.org/abs/1802.04942",
        "role": "Index-code mutual information and aggregated marginal KL.",
    },
    {
        "citation_key": "Joas2026AUTOENCODIX",
        "title": "AUTOENCODIX: a generalized and versatile framework",
        "authors": "Maximilian Josef Joas et al.",
        "year": "2026",
        "identifier": "DOI:10.1038/s43588-025-00916-4",
        "url": "https://doi.org/10.1038/s43588-025-00916-4",
        "role": "AUTOENCODIX coverage and Gaussian total correlation.",
    },
    {
        "citation_key": "RoweDay2019TC",
        "title": "The Sampling Distribution of the Total Correlation for Multivariate Gaussian Random Variables",
        "authors": "Daniel B. Rowe; Cameron R. Day",
        "year": "2019",
        "identifier": "DOI:10.3390/e21100921",
        "url": "https://doi.org/10.3390/e21100921",
        "role": "Gaussian covariance total correlation.",
    },
    {
        "citation_key": "CoverThomas2006Information",
        "title": "Elements of Information Theory",
        "authors": "Thomas M. Cover; Joy A. Thomas",
        "year": "2006",
        "identifier": "Second edition; ISBN:978-0-471-24195-9",
        "url": "https://doi.org/10.1002/047174882X",
        "role": "Differential entropy and Gaussian information-theory identities.",
    },
    {
        "citation_key": "RoyVetterli2007EffectiveRank",
        "title": "The Effective Rank: A Measure of Effective Dimensionality",
        "authors": "Olivier Roy; Martin Vetterli",
        "year": "2007",
        "identifier": "EUSIPCO 2007",
        "url": "https://infoscience.epfl.ch/record/110188",
        "role": "Entropy-based effective rank of the posterior-mean spectrum.",
    },
    {
        "citation_key": "Jeon2023ZADU",
        "title": "ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings",
        "authors": "Hyeon Jeon et al.",
        "year": "2023",
        "identifier": "DOI:10.1109/VIS54172.2023.00048",
        "url": "https://doi.org/10.1109/VIS54172.2023.00048",
        "role": "Software and estimator provenance for label-free geometry metrics.",
    },
    {
        "citation_key": "Yan2023BmVAE",
        "title": "bmVAE: a variational autoencoder method for clustering single-cell mutation data",
        "authors": "Jiaqian Yan; Ming Ma; Zhenhua Yu",
        "year": "2023",
        "identifier": "DOI:10.1093/bioinformatics/btac790",
        "url": "https://doi.org/10.1093/bioinformatics/btac790",
        "role": "Published precedent for analyzing posterior-mean VAE embeddings.",
    },
    {
        "citation_key": "Hira2021MultiOmicsVAE",
        "title": "Integrated multi-omics analysis of ovarian cancer using variational autoencoders",
        "authors": (
            "Muta Tah Hira; M. A. Razzaque; Claudio Angione; James Scrivens; "
            "Saladin Sawan; Mosharraf Sarker"
        ),
        "year": "2021",
        "identifier": "DOI:10.1038/s41598-021-85285-4",
        "url": "https://doi.org/10.1038/s41598-021-85285-4",
        "role": "Published precedent for analyzing sampled VAE latent representations.",
    },
    {
        "citation_key": "Munch2017TDA",
        "title": "A User's Guide to Topological Data Analysis",
        "authors": "Elizabeth Munch",
        "year": "2017",
        "identifier": "DOI:10.18608/jla.2017.42.6",
        "url": "https://doi.org/10.18608/jla.2017.42.6",
        "role": "Persistent homology and Betti-number interpretation.",
    },
    {
        "citation_key": "May2023NormalizedBottleneck",
        "title": "A Normalized Bottleneck Distance on Persistence Diagrams",
        "authors": "Nathan H. May; Bala Krishnamoorthy; Patrick Gambill",
        "year": "2023",
        "identifier": "arXiv:2306.06727",
        "url": "https://arxiv.org/abs/2306.06727",
        "role": "Separate diameter normalization before bottleneck comparison.",
    },
    {
        "citation_key": "Beylier2025CurvatureDR",
        "title": "Curvature as a tool for evaluating dimensionality reduction and estimating intrinsic dimension",
        "authors": "Charlotte Beylier; Parvaneh Joharinad; Jürgen Jost; Nahid Torbati",
        "year": "2025",
        "identifier": "arXiv:2509.13385v1",
        "url": "https://arxiv.org/abs/2509.13385",
        "role": "Optional experimental synthetic sectional-curvature profile comparison.",
    },
    {
        "citation_key": "Arvanitidis2018LatentGeometry",
        "title": "Latent Space Oddity: on the Curvature of Deep Generative Models",
        "authors": "Georgios Arvanitidis; Lars Kai Hansen; Søren Hauberg",
        "year": "2018",
        "identifier": "arXiv:1710.11379v3; ICLR 2018",
        "url": "https://arxiv.org/abs/1710.11379",
        "role": "Optional decoder pullback metric and local latent differential geometry.",
    },
)


RESEARCH_QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "question_id": "Q01_reconstruction",
        "question": "How accurately does VARIX reconstruct held-out observations?",
        "metric_family": "reconstruction",
        "interpretation": "Evaluate MSE, explained variance, and validation/train gap together.",
    },
    {
        "question_id": "Q02_posterior",
        "question": "Which latent dimensions satisfy epsilon-delta posterior collapse?",
        "metric_family": "probabilistic",
        "interpretation": "Use the full conditional-KL matrix before aggregation.",
    },
    {
        "question_id": "Q03_information",
        "question": "How are observation information, coordinate dependence, and marginal prior fit separated?",
        "metric_family": "information",
        "interpretation": "MI, AUTOENCODIX TC, and dimension-wise KL retain distinct estimator identities.",
    },
    {
        "question_id": "Q04_geometry",
        "question": "How well does posterior-mean space preserve input distances and neighborhoods?",
        "metric_family": "geometry",
        "interpretation": "Use aligned X_test versus mu_test on one deterministic subset.",
    },
    {
        "question_id": "Q05_dimensionality",
        "question": "How many intrinsic and dominant linear dimensions occur in X, mu, and z?",
        "metric_family": "representation_dimensionality",
        "interpretation": "Compare TwoNN and MLE estimates and inspect the full posterior-mean PCA spectrum.",
    },
    {
        "question_id": "Q06_topology",
        "question": "How well are multiscale H0 and H1 structures preserved?",
        "metric_family": "topology",
        "interpretation": "Normalize both metric spaces separately before persistent homology.",
    },
)


EXPERIMENT_VARIABLE_ROLES: dict[str, Any] = {
    "independent_factor": {
        "beta": "Configured final regularization weight.",
    },
    "additional_factors": {
        "default_vae_loss": "KL or MMD objective family.",
        "learning_rate": "AdamW learning rate.",
    },
    "fixed_split_factor": {
        "global_seed": "Fixed data-partition seed shared by every run.",
    },
    "replication_factor": {
        "training_seed": "Model initialization and stochastic training seed.",
    },
    "dependent_metric_families": [
        "reconstruction",
        "probabilistic",
        "information",
        "representation_dimensionality",
        "geometry",
        "topology",
    ],
}


BIBTEX = r"""@article{KingmaWelling2014VAE,
  title={Auto-Encoding Variational Bayes},
  author={Kingma, Diederik P. and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2014}
}

@inproceedings{Alemi2018BrokenELBO,
  title={Fixing a Broken ELBO},
  author={Alemi, Alexander A. and Fischer, Ian and Dillon, Joshua V. and Murphy, Kevin},
  booktitle={Proceedings of the 35th International Conference on Machine Learning},
  year={2018},
  eprint={1711.00464},
  archivePrefix={arXiv}
}

@article{Bac2021ScikitDimension,
  title={Scikit-Dimension: A Python Package for Intrinsic Dimension Estimation},
  author={Bac, Jonathan and Mirkes, Evgeny M. and Gorban, Alexander N. and Tyukin, Ivan and Zinovyev, Andrei},
  journal={arXiv preprint arXiv:2109.02596},
  year={2021}
}

@article{Facco2017TwoNN,
  title={Estimating the intrinsic dimension of datasets by a minimal neighborhood information},
  author={Facco, Elena and d'Errico, Maria and Rodriguez, Alex and Laio, Alessandro},
  journal={Scientific Reports},
  volume={7},
  pages={12140},
  year={2017},
  doi={10.1038/s41598-017-11873-y}
}

@inproceedings{Levina2004MLEID,
  title={Maximum Likelihood Estimation of Intrinsic Dimension},
  author={Levina, Elizaveta and Bickel, Peter J.},
  booktitle={Advances in Neural Information Processing Systems 17},
  year={2004}
}

@article{JolliffeCadima2016PCA,
  title={Principal component analysis: a review and recent developments},
  author={Jolliffe, Ian T. and Cadima, Jorge},
  journal={Philosophical Transactions of the Royal Society A},
  volume={374},
  number={2065},
  pages={20150202},
  year={2016},
  doi={10.1098/rsta.2015.0202}
}

@inproceedings{Burda2016IWAE,
  title={Importance Weighted Autoencoders},
  author={Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan},
  booktitle={International Conference on Learning Representations},
  year={2016},
  eprint={1509.00519},
  archivePrefix={arXiv}
}

@inproceedings{Lucas2019PosteriorCollapse,
  title={Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse},
  author={Lucas, James and Tucker, George and Grosse, Roger and Norouzi, Mohammad},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}

@inproceedings{Chen2018BetaTCVAE,
  title={Isolating Sources of Disentanglement in Variational Autoencoders},
  author={Chen, Ricky T. Q. and Li, Xuechen and Grosse, Roger and Duvenaud, David},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}

@article{Joas2026AUTOENCODIX,
  title={AUTOENCODIX: a generalized and versatile framework},
  author={Joas, Maximilian Josef and others},
  journal={Nature Computational Science},
  year={2026},
  doi={10.1038/s43588-025-00916-4}
}

@book{CoverThomas2006Information,
  title={Elements of Information Theory},
  author={Cover, Thomas M. and Thomas, Joy A.},
  edition={2},
  publisher={Wiley-Interscience},
  year={2006},
  doi={10.1002/047174882X}
}

@inproceedings{RoyVetterli2007EffectiveRank,
  title={The Effective Rank: A Measure of Effective Dimensionality},
  author={Roy, Olivier and Vetterli, Martin},
  booktitle={15th European Signal Processing Conference},
  year={2007}
}

@inproceedings{Jeon2023ZADU,
  title={ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings},
  author={Jeon, Hyeon and others},
  booktitle={2023 IEEE Visualization and Visual Analytics},
  year={2023},
  doi={10.1109/VIS54172.2023.00048}
}

@article{Yan2023BmVAE,
  title={bmVAE: a variational autoencoder method for clustering single-cell mutation data},
  author={Yan, Jiaqian and Ma, Ming and Yu, Zhenhua},
  journal={Bioinformatics},
  volume={39},
  number={1},
  year={2023},
  doi={10.1093/bioinformatics/btac790}
}

@article{Hira2021MultiOmicsVAE,
  title={Integrated multi-omics analysis of ovarian cancer using variational autoencoders},
  author={Hira, Muta Tah and Razzaque, M. A. and Angione, Claudio and Scrivens, James and Sawan, Saladin and Sarker, Mosharraf},
  journal={Scientific Reports},
  volume={11},
  pages={6265},
  year={2021},
  doi={10.1038/s41598-021-85285-4}
}

@article{RoweDay2019TC,
  title={The Sampling Distribution of the Total Correlation for Multivariate Gaussian Random Variables},
  author={Rowe, Daniel B. and Day, Cameron R.},
  journal={Entropy},
  volume={21},
  number={10},
  pages={921},
  year={2019},
  doi={10.3390/e21100921}
}

@article{Munch2017TDA,
  title={A User's Guide to Topological Data Analysis},
  author={Munch, Elizabeth},
  journal={Journal of Learning Analytics},
  year={2017},
  doi={10.18608/jla.2017.42.6}
}

@article{May2023NormalizedBottleneck,
  title={A Normalized Bottleneck Distance on Persistence Diagrams},
  author={May, Nathan H. and Krishnamoorthy, Bala and Gambill, Patrick},
  journal={arXiv preprint arXiv:2306.06727},
  year={2023}
}

@article{Beylier2025CurvatureDR,
  title={Curvature as a tool for evaluating dimensionality reduction and estimating intrinsic dimension},
  author={Beylier, Charlotte and Joharinad, Parvaneh and Jost, Jürgen and Torbati, Nahid},
  journal={arXiv preprint arXiv:2509.13385},
  year={2025}
}

@inproceedings{Arvanitidis2018LatentGeometry,
  title={Latent Space Oddity: on the Curvature of Deep Generative Models},
  author={Arvanitidis, Georgios and Hansen, Lars Kai and Hauberg, Søren},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
"""


def save_json(payload: Any, path: Path) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _metric_registry_records() -> list[dict[str, Any]]:
    records = []
    for definition in REGISTRY.values():
        record = asdict(definition)
        record["space_comparison"] = definition.space_comparison.value
        record["representations"] = [
            representation.value for representation in definition.representations
        ]
        records.append(record)
    return sorted(records, key=lambda record: (record["group"], record["name"]))


def save_research_references(output_dir: Path) -> Path:
    """Write all research definitions and citations below ``references/``."""
    target = Path(output_dir) / "references"
    target.mkdir(parents=True, exist_ok=True)
    references = list(SCIENTIFIC_REFERENCES)
    questions = list(RESEARCH_QUESTIONS)
    registry = _metric_registry_records()

    save_json({"references": references}, target / "scientific_references.json")
    save_json({"questions": questions}, target / "research_questions.json")
    save_json(EXPERIMENT_VARIABLE_ROLES, target / "experiment_variable_roles.json")
    save_json({"metrics": registry}, target / "metric_registry.json")
    pd.DataFrame(references).to_csv(target / "scientific_references.csv", index=False)
    pd.DataFrame(questions).to_csv(target / "research_questions.csv", index=False)
    pd.DataFrame(registry).to_csv(target / "metric_registry.csv", index=False)
    (target / "references.bib").write_text(BIBTEX, encoding="utf-8")

    readme = [
        "# Scientific references",
        "",
        "This directory is generated from `beta_vae_experiments.references`.",
        "It is the written scientific mirror of the active absolute-core pipeline.",
        "",
        "## Metric families",
        "",
        "- Reconstruction",
        "- Probabilistic posterior diagnostics",
        "- Information theory",
        "- Representation dimensionality",
        "- Geometry",
        "- Topological data analysis",
        "",
        "Every scalar metric is linked to an exact space comparison, representation,",
        "scientific source, estimator provenance, interpretation, and limitation.",
    ]
    (target / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return target
