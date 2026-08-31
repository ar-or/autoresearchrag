"""H44: LightRAG-style local/global retrieval over the shared graph layer."""

from __future__ import annotations

import os

from experiments.graph_agent_common import make_graph_agent

GRAPH_NAME = os.environ.get("GRAPH_LAYER_NAME", "h44")

create_session, send_message = make_graph_agent("lightrag", GRAPH_NAME)
