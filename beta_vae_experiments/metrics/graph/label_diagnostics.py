"""Labelabhängige Offline-Diagnostik, niemals unüberwachtes Controller-Feature.

Newman (2003): https://doi.org/10.1103/PhysRevE.67.026126
Kategoriale Assortativity vergleicht beobachtetes Mixing mit dem durch
Kantenendpunkt-Häufigkeiten bestimmten Referenzniveau. Fehlende Labels werden
über einen induzierten Teilgraph ausgeschlossen; Anzahl Knoten/Kanten speichern.
Eine Klasse allein ergibt keine definierte Assortativity. Hier keine automatische
Signifikanz-/Kausalitätsaussage; Sex ist zusätzlich durch Cancer konfundierbar.
"""
import numpy as np
import pandas as pd
import networkx as nx


def compute_label_diagnostics(graph, annotations, columns):
    """Return label summaries and ordered mixing rows, outside core scalars."""
    if isinstance(annotations, dict):
        if len(annotations) != 1:
            raise ValueError("Expected one unambiguous annotation table")
        annotations = next(iter(annotations.values()))
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError("Annotations must be a DataFrame indexed by sample ID")
    annotations = annotations.copy()
    annotations.index = annotations.index.astype(str)
    ids = list(map(str, graph.sample_ids))
    if not annotations.index.is_unique or not set(ids) <= set(annotations.index):
        raise ValueError("Annotations must uniquely cover graph IDs")
    annotations = annotations.loc[ids]
    summaries, mixing = [], []
    for column in columns:
        if column not in annotations:
            continue
        labels = annotations[column]
        observed = np.flatnonzero(labels.notna().to_numpy())
        g = graph.undirected.subgraph(observed).copy()
        nx.set_node_attributes(g, {int(i): str(labels.iloc[i]) for i in observed}, "label")
        classes = sorted({g.nodes[i]["label"] for i in g})
        valid = g.number_of_edges() > 0 and len(classes) > 1
        score = float(nx.attribute_assortativity_coefficient(g, "label")) if valid else np.nan
        summaries.append(dict(label=column, value=score, status="ok" if np.isfinite(score) else "undefined",
                              n_nodes=len(g), n_edges=g.number_of_edges(), n_classes=len(classes),
                              n_missing=int(labels.isna().sum()), graph="unweighted_union"))
        if g.number_of_edges():
            matrix = nx.attribute_mixing_matrix(g, "label", mapping={c: i for i, c in enumerate(classes)}, normalized=True)
            for i, source in enumerate(classes):
                for j, target in enumerate(classes):
                    mixing.append(dict(label=column, source_class=source, target_class=target,
                                       edge_end_fraction=float(matrix[i, j])))
    return pd.DataFrame(summaries), pd.DataFrame(mixing)
