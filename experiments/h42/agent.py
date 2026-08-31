"""H42: GraphRAG-style community retrieval over a Memgraph graph layer."""

from __future__ import annotations

import os

from experiments.graph_agent_common import make_graph_agent

GRAPH_NAME = os.environ.get("GRAPH_LAYER_NAME", "h42")

create_session, send_message = make_graph_agent("graphrag", GRAPH_NAME)
