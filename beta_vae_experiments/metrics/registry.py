"""Machine-readable scientific registry for every scalar core metric."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from .contracts import Representation, SpaceComparison


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    group: str
    space_comparison: SpaceComparison
    exact_spaces: tuple[str, ...]
    representations: tuple[Representation, ...]
    required_inputs: tuple[str, ...]
    mathematical_definition: str
    scientific_source: str
    repository_provenance: str
    interpretation: str
    limitations: str
    higher_is_better: bool | None
    unit: str | None
    estimator_name: str


AKVAE = "Kingma and Welling, Auto-Encoding Variational Bayes, arXiv:1312.6114."
LUCAS = (
    "Lucas, Tucker, Grosse, and Norouzi, Don't Blame the ELBO!, "
    "NeurIPS 2019, arXiv:1911.02469."
)
CHEN = (
    "Chen, Li, Grosse, and Duvenaud, Isolating Sources of Disentanglement "
    "in Variational Autoencoders, NeurIPS 2018, arXiv:1802.04942."
)
BURDA = (
    "Burda, Grosse, and Salakhutdinov, Importance Weighted Autoencoders, "
    "ICLR 2016, arXiv:1509.00519, Appendix C."
)
EFFECTIVE_RANK = (
    "Roy and Vetterli, The Effective Rank: A Measure of Effective "
    "Dimensionality, EUSIPCO 2007."
)
AUTOENCODIX = "Joas et al., AUTOENCODIX, DOI:10.1038/s43588-025-00916-4."
ZADU_GEOMETRY = (
    "Jeon et al., ZADU, IEEE VIS 2023, DOI:10.1109/VIS54172.2023.00048; "
    "posterior-mean precedent: Yan, Ma, and Yu, bmVAE, Bioinformatics 2023, "
    "DOI:10.1093/bioinformatics/btac790; sampled-latent precedent: "
    "Hira et al., Scientific Reports 2021, "
    "DOI:10.1038/s41598-021-85285-4."
)
TDA = "Munch, A User's Guide to Topological Data Analysis, DOI:10.18608/jla.2017.42.6."
NORMALIZED_BOTTLENECK = (
    "May, Krishnamoorthy, and Gambill, A Normalized Bottleneck Distance, "
    "arXiv:2306.06727, Definition 2.5."
)
SCIKIT_DIMENSION = (
    "Bac, Mirkes, Gorban, Tyukin, and Zinovyev, Scikit-Dimension: A Python "
    "Package for Intrinsic Dimension Estimation, arXiv:2109.02596."
)
TWONN = (
    "Facco, d'Errico, Rodriguez, and Laio, Estimating the intrinsic dimension "
    "of datasets by a minimal neighborhood information, Scientific Reports "
    "2017, DOI:10.1038/s41598-017-11873-y."
)
MLE_ID = (
    "Levina and Bickel, Maximum Likelihood Estimation of Intrinsic Dimension, "
    "NeurIPS 2004."
)


def _definition(
    name: str,
    group: str,
    comparison: SpaceComparison,
    spaces: tuple[str, ...],
    representations: tuple[Representation, ...],
    inputs: tuple[str, ...],
    formula: str,
    source: str,
    provenance: str,
    interpretation: str,
    limitations: str,
    direction: bool | None,
    estimator: str,
    unit: str | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        group=group,
        space_comparison=comparison,
        exact_spaces=spaces,
        representations=representations,
        required_inputs=inputs,
        mathematical_definition=formula,
        scientific_source=source,
        repository_provenance=provenance,
        interpretation=interpretation,
        limitations=limitations,
        higher_is_better=direction,
        unit=unit,
        estimator_name=estimator,
    )


REGISTRY: dict[str, MetricDefinition] = {}


def _register(definition: MetricDefinition) -> None:
    if definition.name in REGISTRY:
        raise ValueError(f"Duplicate metric registry entry: {definition.name}")
    REGISTRY[definition.name] = definition


for split in ("train", "val", "test"):
    _register(
        _definition(
            f"recon_{split}_mse",
            "reconstruction",
            SpaceComparison.INPUT_VS_RECONSTRUCTION,
            (f"preprocessed model-input X_{split}", f"reconstruction X_hat_{split}"),
            (Representation.INPUT_X, Representation.RECONSTRUCTION_X_HAT),
            (f"x_{split}", f"x_recon_{split}"),
            "Mean of (X_ij - X_hat_ij)^2 over observations and features.",
            AKVAE,
            "NumPy implementation of the cited definition.",
            "Average squared reconstruction error.",
            "Scale-dependent and not directly comparable across differently scaled datasets.",
            False,
            "mean_squared_error",
        )
    )
    _register(
        _definition(
            f"recon_{split}_explained_variance_mean",
            "reconstruction",
            SpaceComparison.INPUT_VS_RECONSTRUCTION,
            (f"preprocessed model-input X_{split}", f"reconstruction X_hat_{split}"),
            (Representation.INPUT_X, Representation.RECONSTRUCTION_X_HAT),
            (f"x_{split}", f"x_recon_{split}"),
            "Mean over samples of 1 - Var(X-X_hat)/Var(X), calculated across features.",
            AUTOENCODIX,
            "Matches sklearn explained_variance_score per observation; AUTOENCODIX reproducibility convention.",
            "Fraction of within-sample feature variance explained by reconstruction.",
            "Undefined for constant samples and sensitive to preprocessing.",
            True,
            "per_sample_explained_variance",
        )
    )

for name, formula in (
    ("recon_val_train_gap", "MSE_val - MSE_train."),
    ("recon_val_train_ratio", "MSE_val / max(MSE_train, epsilon)."),
):
    _register(
        _definition(
            name,
            "reconstruction",
            SpaceComparison.CROSS_SPLIT_RECONSTRUCTION,
            ("training reconstruction errors", "validation reconstruction errors"),
            (Representation.INPUT_X, Representation.RECONSTRUCTION_X_HAT),
            ("x_train", "x_recon_train", "x_val", "x_recon_val"),
            formula,
            "Project-derived diagnostic; no external scientific source required.",
            "Project-derived diagnostic from split-specific MSE.",
            "Quantifies validation versus training reconstruction generalization.",
            "A derived diagnostic, not an independent canonical VAE objective.",
            False,
            "derived_reconstruction_gap",
        )
    )


def _register_family(
    names: tuple[str, ...],
    group: str,
    comparison: SpaceComparison,
    spaces: tuple[str, ...],
    representations: tuple[Representation, ...],
    inputs: tuple[str, ...],
    formula: str,
    source: str,
    provenance: str,
    interpretation: str,
    limitations: str,
    direction: bool | None,
    estimator: str,
    unit: str | None = None,
) -> None:
    for name in names:
        _register(
            _definition(
                name,
                group,
                comparison,
                spaces,
                representations,
                inputs,
                formula,
                source,
                provenance,
                interpretation,
                limitations,
                direction,
                estimator,
                unit,
            )
        )


_register_family(
    (
        "posterior_conditional_kl_total_mean",
        "posterior_conditional_kl_dim_mean",
        "posterior_conditional_kl_dim_median",
        "posterior_conditional_kl_dim_min",
        "posterior_conditional_kl_dim_max",
    ),
    "probabilistic",
    SpaceComparison.POSTERIOR_VS_PRIOR,
    (
        "conditional diagonal-Gaussian posterior q(z_i|x_n)",
        "standard-normal prior p(z_i)",
    ),
    (
        Representation.POSTERIOR_MEAN_MU,
        Representation.POSTERIOR_LOGVAR,
        Representation.PRIOR,
    ),
    ("mu_test", "logvar_test"),
    "KL_ni = 0.5 * (mu_ni^2 + exp(logvar_ni) - logvar_ni - 1).",
    f"{AKVAE} {LUCAS}",
    "Custom analytical diagonal-Gaussian implementation.",
    "Conditional posterior divergence from the prior.",
    "Conditional KL is not aggregated-posterior marginal KL.",
    None,
    "analytical_diagonal_gaussian_kl",
    "nats",
)
_register_family(
    (
        "posterior_collapse_primary_epsilon",
        "posterior_collapse_delta",
        "posterior_collapsed_dimensions_count",
        "posterior_collapsed_dimensions_fraction",
    ),
    "probabilistic",
    SpaceComparison.POSTERIOR_VS_PRIOR,
    (
        "conditional posterior q(z_i|x)",
        "standard-normal prior p(z_i), over empirical x",
    ),
    (Representation.CONDITIONAL_POSTERIOR, Representation.PRIOR),
    ("mu_test", "logvar_test"),
    "A dimension collapses when mean_n[KL_ni < epsilon] >= 1-delta.",
    LUCAS,
    "Custom implementation of Lucas et al. Section 6.2.",
    "Estimates epsilon-delta posterior collapse dimension by dimension.",
    "Thresholds are reporting choices; zero collapse does not establish usefulness.",
    False,
    "epsilon_delta_collapse",
)
_register_family(
    (
        "posterior_active_units_count_eps_0p001",
        "posterior_active_units_fraction_eps_0p001",
        "posterior_active_units_count_eps_0p01",
        "posterior_active_units_fraction_eps_0p01",
        "posterior_active_units_count_eps_0p1",
        "posterior_active_units_fraction_eps_0p1",
    ),
    "probabilistic",
    SpaceComparison.POSTERIOR_ONLY,
    ("empirical posterior-mean representation mu_test",),
    (Representation.POSTERIOR_MEAN_MU,),
    ("mu_test",),
    "A_j=Var_x[mu_j(x)]; unit j is active when A_j>epsilon.",
    BURDA,
    (
        "NumPy population variance with the cited epsilon=0.01 convention "
        "and epsilon in {0.001, 0.1} as project sensitivity thresholds."
    ),
    "Counts posterior-mean coordinates that vary across test observations.",
    (
        "Thresholds are scale-dependent conventions; activity does not prove "
        "semantic usefulness or independence."
    ),
    None,
    "posterior_mean_active_units",
)
_register_family(
    ("latent_coverage_autoencodix",),
    "probabilistic",
    SpaceComparison.LATENT_ONLY,
    ("empirical sampled AUTOENCODIX test latent space z_test",),
    (Representation.SAMPLED_LATENT_Z,),
    ("z_test",),
    "B=floor(n^(1/d)); occupied joint B^d cells divided by B^d.",
    AUTOENCODIX,
    "Direct call to varix.visualizer._coverage_calc on the trained pipeline visualizer.",
    "Measures joint occupancy of the observed latent space.",
    "Depends on sample count, binning, dimensionality, and outliers; not prior fit or semantic quality.",
    True,
    "autoencodix_joint_occupied_cell_coverage",
)
_register_family(
    (
        "posterior_entropy_total_mean",
        "posterior_entropy_total_median",
        "posterior_entropy_total_p95",
        "posterior_entropy_per_dimension_mean",
    ),
    "probabilistic",
    SpaceComparison.POSTERIOR_ONLY,
    ("conditional diagonal-Gaussian posterior q(z|x)",),
    (Representation.POSTERIOR_LOGVAR, Representation.CONDITIONAL_POSTERIOR),
    ("logvar_test",),
    "H[q(z|x)] = 0.5 * sum_i(1 + log(2*pi) + logvar_i(x)).",
    "Cover and Thomas, Elements of Information Theory.",
    "Custom analytical diagonal-Gaussian entropy implementation.",
    "Average conditional posterior spread.",
    "Not calibration, epistemic uncertainty, or representation quality by itself.",
    None,
    "analytical_diagonal_gaussian_entropy",
    "nats",
)

_register_family(
    ("info_mutual_information_nats",),
    "information",
    SpaceComparison.INPUT_VS_LATENT,
    (
        "empirical observation identity X",
        "stochastic latent representation under q(x,z)",
    ),
    (Representation.CONDITIONAL_POSTERIOR, Representation.AGGREGATED_POSTERIOR),
    ("mu_test", "logvar_test", "z_test"),
    "E_q(x,z)[log q(z|x) - log q(z)].",
    CHEN,
    "Project beta-TCVAE-style mixture-density estimator.",
    "Index-code mutual information between observations and stochastic latents.",
    "Finite-sample mixture estimator is biased; larger is not universally better.",
    None,
    "beta_tcvae_mixture_density",
    "nats",
)
_register_family(
    ("info_total_correlation_autoencodix",),
    "information",
    SpaceComparison.AGGREGATED_POSTERIOR_ONLY,
    (
        "joint sampled AUTOENCODIX latent coordinates",
        "product of their Gaussian marginals",
    ),
    (Representation.SAMPLED_LATENT_Z, Representation.AGGREGATED_POSTERIOR),
    ("z_test",),
    "0.5 * (sum_i log Sigma_ii - log det Sigma).",
    f"{AUTOENCODIX} Rowe and Day, Entropy 2019, 21, 921.",
    "Direct call to varix.visualizer._total_correlation on the trained pipeline visualizer.",
    "Gaussian covariance total correlation among latent coordinates.",
    "Captures covariance dependence only and may be unstable near singularity.",
    False,
    "autoencodix_gaussian_covariance_tc",
    "nats",
)
_register_family(
    ("info_total_correlation_beta_tcvae_mixture_nats",),
    "information",
    SpaceComparison.AGGREGATED_POSTERIOR_ONLY,
    (
        "aggregated posterior q(z)",
        "product of aggregated posterior marginals q(z_i)",
    ),
    (Representation.AGGREGATED_POSTERIOR,),
    ("mu_test", "logvar_test", "z_test"),
    "E_q(z)[log q(z) - sum_i log q(z_i)].",
    CHEN,
    "Project beta-TCVAE-style finite-mixture density estimator.",
    "Non-factorial dependence among latent coordinates under the aggregated posterior.",
    (
        "Finite-sample mixture estimator is biased and computationally quadratic; "
        "it is not interchangeable with AUTOENCODIX Gaussian covariance TC."
    ),
    False,
    "beta_tcvae_mixture_density",
    "nats",
)
_register_family(
    ("info_dimension_wise_kl_nats",),
    "information",
    SpaceComparison.POSTERIOR_MARGINALS_VS_PRIOR,
    ("aggregated posterior marginals q(z_i)", "standard-normal prior marginals p(z_i)"),
    (Representation.AGGREGATED_POSTERIOR_MARGINALS, Representation.PRIOR),
    ("mu_test", "logvar_test", "z_test"),
    "E_q(z)[sum_i log q(z_i) - log p(z_i)].",
    CHEN,
    "Project beta-TCVAE-style mixture-density estimator.",
    "Marginal aggregated-posterior divergence from the prior.",
    "Distinct from E_x KL(q(z_i|x)||p(z_i)); finite-sample estimator is biased.",
    False,
    "beta_tcvae_mixture_density",
    "nats",
)

_register_family(
    ("info_decomposition_residual_nats",),
    "information",
    SpaceComparison.POSTERIOR_VS_PRIOR,
    ("conditional posterior and prior", "beta-TCVAE decomposition components"),
    (
        Representation.CONDITIONAL_POSTERIOR,
        Representation.AGGREGATED_POSTERIOR,
        Representation.PRIOR,
    ),
    ("mu_test", "logvar_test", "z_test"),
    "R - (I(X;Z) + TC(Z) + sum_j KL(q(z_j)||p(z_j))).",
    CHEN,
    (
        "Project numerical consistency diagnostic derived from the theoretical "
        "identity in Chen et al.; it is not a separate metric proposed there."
    ),
    "Numerical discrepancy between analytical rate and its estimated decomposition.",
    (
        "The analytical rate and finite-sample mixture components use different "
        "Monte Carlo estimators, so a non-zero residual is expected."
    ),
    None,
    "beta_tcvae_decomposition_consistency_residual",
    "nats",
)

_register_family(
    ("info_decomposition_rate_reference_nats",),
    "information",
    SpaceComparison.POSTERIOR_VS_PRIOR,
    ("conditional diagonal-Gaussian posterior", "standard-normal prior"),
    (
        Representation.CONDITIONAL_POSTERIOR,
        Representation.PRIOR,
    ),
    ("mu_test", "logvar_test"),
    "E_x[KL(q(z|x)||p(z))] on the exact subset used by the mixture estimator.",
    CHEN,
    "Project analytical diagonal-Gaussian rate on the estimator-selected subset.",
    "Rate reference used only to calculate the decomposition consistency residual.",
    "Subset-specific diagnostic; it can differ from the full-test analytical rate.",
    None,
    "analytical_rate_on_information_estimator_subset",
    "nats",
)

_register_family(
    ("geometry_effective_rank_mu",),
    "geometry",
    SpaceComparison.LATENT_ONLY,
    ("column-centered empirical posterior-mean test representation mu_test",),
    (Representation.POSTERIOR_MEAN_MU,),
    ("mu_test",),
    "exp(-sum_i p_i log p_i), where p_i=sigma_i/sum_j sigma_j.",
    EFFECTIVE_RANK,
    "NumPy SVD of column-centered mu_test using normalized singular values.",
    "Continuous effective number of represented variation directions.",
    (
        "Does not establish semantic usefulness, disentanglement, or "
        "input-geometry preservation; zero-centered rank is reported as zero."
    ),
    None,
    "roy_vetterli_effective_rank_centered_mu",
)

for representation_name, representation, comparison, required_input in (
    ("x", Representation.INPUT_X, SpaceComparison.INPUT_ONLY, "x_test"),
    (
        "mu",
        Representation.POSTERIOR_MEAN_MU,
        SpaceComparison.LATENT_ONLY,
        "mu_test",
    ),
    (
        "z",
        Representation.SAMPLED_LATENT_Z,
        SpaceComparison.LATENT_ONLY,
        "z_test",
    ),
):
    _register(
        _definition(
            f"dimensionality_{representation_name}_twonn",
            "representation_dimensionality",
            comparison,
            (f"aligned test representation {representation_name}",),
            (representation,),
            (required_input,),
            "TwoNN intrinsic-dimension estimate from first/second-neighbor distance ratios.",
            f"{TWONN} {SCIKIT_DIMENSION}",
            "scikit-dimension 0.3.4 skdim.id.TwoNN public estimator API.",
            "Estimated global nonlinear degrees of freedom in the representation.",
            "Sensitive to sampling density, noise, duplicate points, and finite sample size.",
            None,
            "skdim_id_twonn",
        )
    )
    _register(
        _definition(
            f"dimensionality_{representation_name}_mle",
            "representation_dimensionality",
            comparison,
            (f"aligned test representation {representation_name}",),
            (representation,),
            (required_input,),
            "Neighborhood-based maximum-likelihood intrinsic-dimension estimate.",
            f"{MLE_ID} {SCIKIT_DIMENSION}",
            (
                "scikit-dimension 0.3.4 skdim.id.MLE public estimator API "
                "with an explicit configured fit(n_neighbors=...)."
            ),
            "Estimated global nonlinear degrees of freedom in the representation.",
            "Sensitive to neighborhood scale, sampling density, noise, and manifold assumptions.",
            None,
            "skdim_id_mle",
        )
    )
for space in ("input", "latent"):
    comparison = (
        SpaceComparison.INPUT_ONLY
        if space == "input"
        else SpaceComparison.LATENT_TOPOLOGY_ONLY
    )
    representation = (
        Representation.INPUT_PERSISTENCE_DIAGRAM
        if space == "input"
        else Representation.LATENT_PERSISTENCE_DIAGRAM
    )
    for dim in (0, 1):
        _register_family(
            (
                f"topology_{space}_h{dim}_finite_feature_count",
                f"topology_{space}_h{dim}_total_persistence",
                f"topology_{space}_h{dim}_max_persistence",
                f"topology_{space}_h{dim}_median_persistence",
            ),
            "topology",
            comparison,
            (f"diameter-normalized {space} metric space",),
            (representation,),
            ("x_test" if space == "input" else "mu_test",),
            "Summary of finite lifetimes d_i-b_i in the H0 or H1 persistence diagram.",
            TDA,
            "ripser Vietoris-Rips persistence diagrams.",
            "Characterizes persistent features within one metric space.",
            "A single-space descriptor is not a topology-preservation metric.",
            None,
            "vietoris_rips_persistence_summary",
        )

for dim in (0, 1):
    _register(
        _definition(
            f"topology_h{dim}_normalized_bottleneck",
            "topology",
            SpaceComparison.INPUT_TOPOLOGY_VS_LATENT_TOPOLOGY,
            (
                "H0/H1 diagram of separately diameter-normalized input metric space",
                "H0/H1 diagram of separately diameter-normalized posterior-mean latent metric space",
            ),
            (
                Representation.INPUT_PERSISTENCE_DIAGRAM,
                Representation.LATENT_PERSISTENCE_DIAGRAM,
            ),
            ("x_test", "mu_test"),
            "Bottleneck distance after dividing each metric-space distance matrix by its own diameter.",
            NORMALIZED_BOTTLENECK,
            "ripser diagrams and persim.bottleneck finite-diagram comparison.",
            "Scale-invariant discrepancy between persistent topological signatures.",
            "Depends on metric and sample; emphasizes the largest unmatched feature.",
            False,
            "separate_diameter_normalized_bottleneck",
        )
    )

for dim in (0, 1):
    _register(
        _definition(
            f"topology_h{dim}_normalized_wasserstein",
            "topology",
            SpaceComparison.INPUT_TOPOLOGY_VS_LATENT_TOPOLOGY,
            (
                "H0/H1 diagram of separately diameter-normalized input metric space",
                "H0/H1 diagram of separately diameter-normalized posterior-mean latent metric space",
            ),
            (
                Representation.INPUT_PERSISTENCE_DIAGRAM,
                Representation.LATENT_PERSISTENCE_DIAGRAM,
            ),
            ("x_test", "mu_test"),
            "Wasserstein optimal-matching cost between finite persistence diagrams after dividing each metric-space distance matrix by its own diameter.",
            TDA,
            "ripser diagrams and persim.wasserstein finite-diagram comparison.",
            "Scale-invariant aggregate discrepancy across persistent topological features.",
            "Depends on metric and sample and can accumulate many small or noisy feature discrepancies; essential classes are excluded.",
            False,
            "separate_diameter_normalized_wasserstein",
        )
    )

_register_family(
    ("topology_input_diameter", "topology_latent_diameter"),
    "topology",
    SpaceComparison.INPUT_TOPOLOGY_VS_LATENT_TOPOLOGY,
    ("aligned input metric space", "aligned posterior-mean latent metric space"),
    (Representation.INPUT_DISTANCE_MATRIX, Representation.LATENT_DISTANCE_MATRIX),
    ("x_test", "mu_test"),
    "Maximum pairwise distance before separate diameter normalization.",
    NORMALIZED_BOTTLENECK,
    "SciPy pairwise-distance matrix maximum.",
    "Records the normalization scale of each metric space.",
    "Descriptive only and sensitive to outliers.",
    None,
    "metric_space_diameter",
)


def get_metric_definition(name: str) -> MetricDefinition:
    """Return a scalar definition, creating registry-backed dynamic-k entries."""
    if name in REGISTRY:
        return REGISTRY[name]
    zadu_match = re.fullmatch(
        r"geometry_(mu|z)_(trustworthiness|continuity|mrre_false|"
        r"mrre_missing|lcmc|neighbor_dissimilarity)_zadu_k(\d+)",
        name,
    )
    zadu_stress_match = re.fullmatch(
        r"geometry_(mu|z)_(scale_normalized_stress|non_metric_stress)_zadu",
        name,
    )
    if zadu_match or zadu_stress_match:
        if zadu_match:
            space, metric, k = zadu_match.groups()
            if metric == "neighbor_dissimilarity":
                formula = (
                    "ZADU neighbor dissimilarity between the input and latent "
                    f"shared-nearest-neighbor graphs at k={k}."
                )
                interpretation = (
                    "Shared-nearest-neighbor graph disagreement; zero means "
                    "identical shared-neighbor structure."
                )
                direction = False
            else:
                formula = f"ZADU {metric} input-versus-latent criterion at k={k}."
                interpretation = f"{metric} geometry preservation at k={k}."
                direction = True
        else:
            assert zadu_stress_match is not None
            space, metric = zadu_stress_match.groups()
            if metric == "scale_normalized_stress":
                formula = (
                    "ZADU stress after fitting the optimal global distance "
                    "scale between input and latent spaces."
                )
                interpretation = (
                    "Pairwise-distance distortion invariant to one global scale."
                )
            else:
                formula = (
                    "ZADU/Kruskal non-metric stress after monotone isotonic "
                    "fitting of input and latent pairwise distances."
                )
                interpretation = (
                    "Distortion of pairwise-distance ordering after allowing a "
                    "monotone distance transformation."
                )
            direction = False
        latent_name = (
            "posterior-mean latent space mu_test"
            if space == "mu"
            else "persisted sampled latent space z_test"
        )
        representation = (
            Representation.POSTERIOR_MEAN_MU
            if space == "mu"
            else Representation.SAMPLED_LATENT_Z
        )
        definition = _definition(
            name,
            "geometry",
            SpaceComparison.INPUT_VS_LATENT,
            ("aligned preprocessed model-input test space X_test", latent_name),
            (Representation.INPUT_X, representation),
            ("x_test", "mu_test" if space == "mu" else "z_test"),
            formula,
            ZADU_GEOMETRY,
            (
                "ZADU 0.4.2 public class API: "
                "ZADU(specifications, X).measure(embedding)."
            ),
            interpretation,
            (
                "Depends on preprocessing, Euclidean geometry, sample subset, "
                "and k; z additionally contains posterior sampling variation."
            ),
            direction,
            f"zadu_{metric}",
        )
        REGISTRY[name] = definition
        return definition
    raise KeyError(f"No scientific registry entry for scalar metric {name!r}")


def metric_catalog_frame() -> pd.DataFrame:
    rows = []
    for definition in REGISTRY.values():
        row = asdict(definition)
        row["space_comparison"] = definition.space_comparison.value
        row["representations"] = [value.value for value in definition.representations]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "name"]).reset_index(drop=True)
