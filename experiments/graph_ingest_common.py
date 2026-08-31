"""Shared ingest entrypoint for graph-layer experiments."""

from __future__ import annotations

from experiments.graph_layer import (
    build_hotpot_graph_layer,
    clone_graph_layer,
    graph_layer_stats,
)


def run_graph_ingest(graph_name: str, variant: str) -> None:
    existing = graph_layer_stats(graph_name)
    if existing["documents"] > 0:
        print(
            "Graph layer already present",
            f"graph={graph_name}",
            f"variant={variant}",
            f"documents={existing['documents']}",
            f"entities={existing['entities']}",
            f"communities={existing['communities']}",
            f"related_edges={existing['related_edges']}",
        )
        return

    suffix = ""
    if "_" in graph_name:
        suffix = graph_name[graph_name.index("_") :]

    for candidate in tuple(f"{base}{suffix}" for base in ("h42", "h44", "h45", "h46")):
        if candidate == graph_name:
            continue
        candidate_stats = graph_layer_stats(candidate)
        if candidate_stats["documents"] <= 0:
            continue
        cloned = clone_graph_layer(candidate, graph_name, variant=variant)
        print(
            "Cloned graph layer",
            f"source={candidate}",
            f"graph={graph_name}",
            f"variant={variant}",
            f"documents={cloned['documents']}",
            f"entities={cloned['entities']}",
            f"communities={cloned['communities']}",
            f"related_edges={cloned['related_edges']}",
        )
        return

    stats = build_hotpot_graph_layer(graph_name=graph_name, variant=variant)
    print(
        "Built graph layer",
        f"graph={graph_name}",
        f"variant={variant}",
        f"documents={stats['documents']}",
        f"entities={stats['entities']}",
        f"communities={stats['communities']}",
        f"related_edges={stats['related_edges']}",
    )
