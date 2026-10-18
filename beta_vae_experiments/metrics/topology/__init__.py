"""Topologische Datenanalyse (TDA): öffentliche Schnittstelle.

Für die Masterarbeit:
Persistente Homologie untersucht, wie Zusammenhang und Schleifen beim
Vergrößern einer Abstandsschwelle entstehen und verschwinden. Wir verwenden
Vietoris–Rips-Filtrationen auf paarweisen euklidischen Distanzen und berechnen
H0/H1. Referenz: https://arxiv.org/abs/1207.3885 .
Deskriptive Topologie und Repräsentationserhaltung beantworten verschiedene
Fragen und werden daher in zwei fachlichen Modulen getrennt.
Bestehende öffentliche Imports bleiben erhalten.

preservation.py: paired X/encoding Rips diagrams and preservation distances.
latent_only.py: deskriptive Topologie einzelner Repräsentationen.
landscape.py: optional functional descriptors.
"""
from .preservation import (
    TopologyMetricConfig,
    compute_h0_signature_preservation,
    SIGNATURE_METRIC_NAME,
    validate_paired_dissimilarity_matrices,
    compute_persistence_diagrams_h0_h1,
    summarize_persistence_diagrams,
    compute_distance_scale_diagnostics,
    compute_normalized_wasserstein_h0_h1,
    compute_normalized_bottleneck_h0_h1,
    compute_persistence_entropy,
    compute_betti_curve_artifacts,
    run_topology_metrics_pipeline,
    RIPSER_CONFIG,
    WASSERSTEIN_METRIC_NAMES,
    BOTTLENECK_METRIC_NAMES,
    ENTROPY_METRIC_NAMES,
    PERSISTENCE_METRIC_NAMES,
    TOPOLOGY_METRIC_NAMES
)
