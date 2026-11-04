"""Metric implementations with lazy exports for optional analysis modules."""

__all__ = ["TopologyMetricConfig", "GraphMetricConfig"]


def __getattr__(name):
    if name == "TopologyMetricConfig":
        from .topology import TopologyMetricConfig
        value = TopologyMetricConfig
    elif name == "GraphMetricConfig":
        from .graph.graph_metrics_pipeline import GraphMetricConfig
        value = GraphMetricConfig
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
