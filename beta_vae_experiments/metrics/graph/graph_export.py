"""Aligned node/edge CSV artifacts and optional NetworkX GraphML export.

Nonfinite node values remain in CSV, omitted from GraphML for interoperability.
See graph_metrics.md. Labels, if supplied, are external metadata only.
"""
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd


def export_graph(graph, nodes, edges, directory, graphml=False, sample_metadata=None):
    nodes = dict(nodes)
    if isinstance(sample_metadata, dict):
        if len(sample_metadata) != 1:
            raise ValueError("Expected one unambiguous annotation table")
        sample_metadata = next(iter(sample_metadata.values()))
    if sample_metadata is not None:
        if not isinstance(sample_metadata, pd.DataFrame):
            raise ValueError("External metadata must be a DataFrame indexed by sample ID")
        table = sample_metadata.copy()
        table.index = table.index.astype(str)
        if not table.index.is_unique or not set(map(str, graph.sample_ids)) <= set(table.index):
            raise ValueError("External metadata IDs do not cover graph cohort uniquely")
        table = table.loc[list(map(str, graph.sample_ids))]
        for name in table.columns:
            nodes['external_'+str(name)] = table[name].to_numpy()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": graph.sample_ids, **nodes}).to_csv(directory/"nodes.csv", index=False)
    g = graph.undirected.copy()
    for name, values in nodes.items():
        if len(values) != len(g):
            raise ValueError(f"Unaligned node artifact: {name}")
        for i, value in enumerate(values):
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                continue
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            if not isinstance(value, (int, float)) or np.isfinite(value):
                g.nodes[i][name] = value
    for name, values in edges.items():
        nx.set_edge_attributes(g, values, name)
    records = [{"source_sample_id": str(graph.sample_ids[i]), "target_sample_id": str(graph.sample_ids[j]), **data}
               for i, j, data in g.edges(data=True)]
    pd.DataFrame(records).to_csv(directory/"edges.csv", index=False)
    if graphml:
        nx.write_graphml(g, directory/"union.graphml")
        directed = graph.directed.copy()
        nx.set_node_attributes(directed, dict(g.nodes(data=True)))
        for i, j in directed.edges:
            directed[i][j].update({name: value for name, value in g[i][j].items() if name != 'distance'})
        nx.write_graphml(directed, directory/"directed.graphml")
