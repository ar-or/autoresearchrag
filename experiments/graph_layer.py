"""Shared Memgraph-backed graph layer for graph retrieval experiments."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import requests
from neo4j import GraphDatabase

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")
GRAPH_BUILD_LIMIT = int(os.environ.get("GRAPH_BUILD_LIMIT", "0") or "0")
MAX_ENTITY_MENTIONS_PER_DOC = int(os.environ.get("GRAPH_MAX_ENTITY_MENTIONS", "16"))
MAX_COMMUNITY_TITLES = int(os.environ.get("GRAPH_MAX_COMMUNITY_TITLES", "6"))
MAX_DOCS_PER_ENTITY_LINK = int(os.environ.get("GRAPH_MAX_DOCS_PER_ENTITY_LINK", "32"))
ELASTIC_URL = os.environ.get("ES_HOST", "http://localhost:9200")
ELASTIC_INDEX = os.environ.get("ES_INDEX", "feature_h04_structure_aware")
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY", "")
ELASTIC_TIMEOUT_S = float(os.environ.get("ELASTIC_TIMEOUT_S", "30"))

ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,3})\b"
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}


@dataclass(frozen=True)
class GraphDoc:
    document_id: str
    title: str
    text: str
    entity_names: tuple[str, ...]


@dataclass(frozen=True)
class GraphSnapshot:
    graph_name: str
    documents: dict[str, dict[str, object]]
    entities: dict[str, dict[str, object]]
    communities: dict[str, dict[str, object]]
    groups: dict[str, dict[str, object]]
    doc_entities: dict[str, set[str]]
    entity_docs: dict[str, set[str]]
    community_docs: dict[str, set[str]]
    community_entities: dict[str, set[str]]
    group_entities: dict[str, set[str]]
    graph: nx.Graph


def _graph_auth():
    if MEMGRAPH_USERNAME:
        return (MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)
    return None


def _driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=_graph_auth())


def _es_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if ELASTIC_API_KEY:
        headers["Authorization"] = f"ApiKey {ELASTIC_API_KEY}"
    return headers


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def _normalize_entity(text: str) -> str:
    tokens = _normalize_tokens(text)
    return " ".join(tokens)


def _entity_group_key(entity_name: str) -> str:
    tokens = _normalize_tokens(entity_name)
    return tokens[0] if tokens else "misc"


def _primary_group_key(entity_names: tuple[str, ...]) -> str:
    counts: Counter[str] = Counter()
    for entity_name in entity_names:
        counts[_entity_group_key(entity_name)] += 1
    if not counts:
        return "misc"
    return counts.most_common(1)[0][0]


def _extract_entities(title: str, text: str) -> list[str]:
    candidates = {title.strip()}
    candidates.update(match.group(0).strip() for match in ENTITY_PATTERN.finditer(text))
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        normalized = _normalize_entity(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(candidate)
    return cleaned[:MAX_ENTITY_MENTIONS_PER_DOC]


def pagerank_scores(
    graph: nx.Graph,
    personalization: dict[str, float] | None = None,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1.0e-6,
) -> dict[str, float]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}

    node_count = len(nodes)
    if personalization:
        total = sum(max(value, 0.0) for value in personalization.values()) or 1.0
        base = {node: max(personalization.get(node, 0.0), 0.0) / total for node in nodes}
    else:
        base = {node: 1.0 / node_count for node in nodes}

    scores = dict(base)
    for _ in range(max_iter):
        updated = {node: (1.0 - alpha) * base[node] for node in nodes}
        dangling_mass = 0.0
        for node in nodes:
            neighbors = list(graph.neighbors(node))
            score = scores[node]
            if not neighbors:
                dangling_mass += score
                continue
            total_weight = sum(float(graph[node][neighbor].get("weight", 1.0)) for neighbor in neighbors) or 1.0
            for neighbor in neighbors:
                weight = float(graph[node][neighbor].get("weight", 1.0))
                updated[neighbor] += alpha * score * (weight / total_weight)
        if dangling_mass:
            for node in nodes:
                updated[node] += alpha * dangling_mass * base[node]
        diff = sum(abs(updated[node] - scores[node]) for node in nodes)
        scores = updated
        if diff < tol:
            break
    return scores


def _iter_es_hits(index_name: str):
    response = requests.post(
        f"{ELASTIC_URL}/{index_name}/_search?scroll=2m",
        headers=_es_headers(),
        json={
            "size": 1000,
            "sort": ["_doc"],
            "query": {"match_all": {}},
            "_source": ["document_id", "title", "doc_name", "source", "text", "content", "pageContent", "chunk_index"],
        },
        timeout=ELASTIC_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    scroll_id = payload.get("_scroll_id")
    try:
        hits = payload.get("hits", {}).get("hits", [])
        while hits:
            for hit in hits:
                yield hit
            if not scroll_id:
                break
            response = requests.post(
                f"{ELASTIC_URL}/_search/scroll",
                headers=_es_headers(),
                json={"scroll": "2m", "scroll_id": scroll_id},
                timeout=ELASTIC_TIMEOUT_S,
            )
            response.raise_for_status()
            payload = response.json()
            scroll_id = payload.get("_scroll_id", scroll_id)
            hits = payload.get("hits", {}).get("hits", [])
    finally:
        if scroll_id:
            try:
                requests.delete(
                    f"{ELASTIC_URL}/_search/scroll",
                    headers=_es_headers(),
                    json={"scroll_id": [scroll_id]},
                    timeout=ELASTIC_TIMEOUT_S,
                )
            except Exception:
                pass


def _load_index_documents(limit: int = 0, index_name: str | None = None) -> list[GraphDoc]:
    target_index = index_name or ELASTIC_INDEX
    grouped: dict[str, dict[str, object]] = {}

    for hit in _iter_es_hits(target_index):
        source = hit.get("_source", {})
        document_id = str(source.get("document_id") or hit.get("_id") or "")
        if not document_id:
            continue
        title = str(
            source.get("title")
            or source.get("doc_name")
            or source.get("source")
            or document_id
        )
        text = str(
            source.get("text")
            or source.get("content")
            or source.get("pageContent")
            or ""
        ).strip()
        if not text:
            continue
        row = grouped.setdefault(
            document_id,
            {"title": title, "chunks": []},
        )
        chunk_index = int(source.get("chunk_index", len(row["chunks"])) or 0)
        cast_chunks = row["chunks"]
        assert isinstance(cast_chunks, list)
        cast_chunks.append((chunk_index, text))

    items = list(grouped.items())
    if limit > 0:
        items = items[:limit]

    docs: list[GraphDoc] = []
    for document_id, row in items:
        chunks = sorted(row["chunks"], key=lambda item: (item[0], item[1]))
        text = "\n".join(chunk_text for _, chunk_text in chunks).strip()
        if not text:
            continue
        title = str(row["title"])
        docs.append(
            GraphDoc(
                document_id=document_id,
                title=title,
                text=text,
                entity_names=tuple(_extract_entities(title, text)),
            )
        )
    return docs


def _build_graph_records(
    documents: list[GraphDoc],
    graph_name: str,
    variant: str,
) -> dict[str, list[dict[str, object]]]:
    doc_graph = nx.Graph()
    entity_docs: dict[str, set[str]] = defaultdict(set)
    entity_display_names: dict[str, str] = {}
    document_rows: list[dict[str, object]] = []
    mention_rows: list[dict[str, object]] = []

    for doc in documents:
        doc_graph.add_node(doc.document_id, kind="document", title=doc.title)
        document_rows.append(
            {
                "graph": graph_name,
                "variant": variant,
                "document_id": doc.document_id,
                "title": doc.title,
                "text": doc.text,
                "entity_count": len(doc.entity_names),
            }
        )
        for entity_name in doc.entity_names:
            norm = _normalize_entity(entity_name)
            if not norm:
                continue
            entity_docs[norm].add(doc.document_id)
            entity_display_names.setdefault(norm, entity_name)
            mention_rows.append(
                {
                    "graph": graph_name,
                    "document_id": doc.document_id,
                    "entity_norm": norm,
                }
            )

    pagerank = pagerank_scores(doc_graph) if doc_graph.number_of_nodes() else {}
    for row in document_rows:
        row["pagerank"] = float(pagerank.get(str(row["document_id"]), 0.0))

    if doc_graph.number_of_edges():
        try:
            communities = list(nx.community.louvain_communities(doc_graph, weight="weight"))
        except Exception:
            communities = list(nx.community.greedy_modularity_communities(doc_graph, weight="weight"))
    else:
        grouped_communities: dict[str, set[str]] = defaultdict(set)
        for doc in documents:
            grouped_communities[_primary_group_key(doc.entity_names)].add(doc.document_id)
        communities = list(grouped_communities.values())

    doc_lookup = {doc.document_id: doc for doc in documents}
    entity_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    entity_group_rows: list[dict[str, object]] = []
    community_rows: list[dict[str, object]] = []
    community_doc_rows: list[dict[str, object]] = []
    community_entity_rows: list[dict[str, object]] = []
    seen_groups: set[str] = set()

    for norm, doc_ids in entity_docs.items():
        group_key = _entity_group_key(entity_display_names[norm])
        entity_rows.append(
            {
                "graph": graph_name,
                "variant": variant,
                "norm": norm,
                "name": entity_display_names[norm],
                "frequency": len(doc_ids),
                "group_key": group_key,
            }
        )
        entity_group_rows.append(
            {
                "graph": graph_name,
                "entity_norm": norm,
                "group_key": group_key,
            }
        )
        if group_key not in seen_groups:
            group_rows.append(
                {
                    "graph": graph_name,
                    "variant": variant,
                    "group_key": group_key,
                    "label": group_key.title(),
                }
            )
            seen_groups.add(group_key)

    for index, community in enumerate(communities):
        doc_ids = sorted(community)
        community_counter: Counter[str] = Counter()
        for doc_id in doc_ids:
            for entity_name in doc_lookup[doc_id].entity_names:
                norm = _normalize_entity(entity_name)
                if norm:
                    community_counter[norm] += 1
        top_entities = [entity_display_names[norm] for norm, _ in community_counter.most_common(6)]
        top_titles = [doc_lookup[doc_id].title for doc_id in doc_ids[:MAX_COMMUNITY_TITLES]]
        summary = (
            f"Community {index} centers on {', '.join(top_entities[:4]) or 'linked topics'}. "
            f"Representative pages: {', '.join(top_titles[:4])}."
        )
        community_id = f"{graph_name}:community:{index}"
        community_rows.append(
            {
                "graph": graph_name,
                "variant": variant,
                "community_id": community_id,
                "summary": summary,
                "size": len(doc_ids),
            }
        )
        for doc_id in doc_ids:
            community_doc_rows.append(
                {
                    "graph": graph_name,
                    "community_id": community_id,
                    "document_id": doc_id,
                }
            )
        for norm, _ in community_counter.most_common(8):
            community_entity_rows.append(
                {
                    "graph": graph_name,
                    "community_id": community_id,
                    "entity_norm": norm,
                }
            )

    return {
        "documents": document_rows,
        "entities": entity_rows,
        "groups": group_rows,
        "mentions": mention_rows,
        "entity_groups": entity_group_rows,
        "communities": community_rows,
        "community_docs": community_doc_rows,
        "community_entities": community_entity_rows,
        "related_docs": [],
    }


def _run_write_batches(session, rows: list[dict[str, object]], query: str, batch_size: int = 500) -> None:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if batch:
            session.run(query, rows=batch)


def build_hotpot_graph_layer(graph_name: str, variant: str) -> dict[str, int]:
    documents = _load_index_documents(limit=GRAPH_BUILD_LIMIT)
    records = _build_graph_records(documents, graph_name=graph_name, variant=variant)

    with _driver() as driver:
        with driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE n.graph = $graph
                DETACH DELETE n
                """,
                graph=graph_name,
            )
            _run_write_batches(
                session,
                records["documents"],
                """
                UNWIND $rows AS row
                CREATE (:Document {
                  graph: row.graph,
                  variant: row.variant,
                  document_id: row.document_id,
                  title: row.title,
                  text: row.text,
                  entity_count: row.entity_count,
                  pagerank: row.pagerank
                })
                """,
            )
            _run_write_batches(
                session,
                records["entities"],
                """
                UNWIND $rows AS row
                CREATE (:Entity {
                  graph: row.graph,
                  variant: row.variant,
                  norm: row.norm,
                  name: row.name,
                  frequency: row.frequency,
                  group_key: row.group_key
                })
                """,
            )
            _run_write_batches(
                session,
                records["groups"],
                """
                UNWIND $rows AS row
                CREATE (:EntityGroup {
                  graph: row.graph,
                  variant: row.variant,
                  group_key: row.group_key,
                  label: row.label
                })
                """,
            )
            _run_write_batches(
                session,
                records["communities"],
                """
                UNWIND $rows AS row
                CREATE (:Community {
                  graph: row.graph,
                  variant: row.variant,
                  community_id: row.community_id,
                  summary: row.summary,
                  size: row.size
                })
                """,
            )
            _run_write_batches(
                session,
                records["mentions"],
                """
                UNWIND $rows AS row
                MATCH (d:Document {graph: row.graph, document_id: row.document_id})
                MATCH (e:Entity {graph: row.graph, norm: row.entity_norm})
                CREATE (d)-[:MENTIONS]->(e)
                """,
            )
            _run_write_batches(
                session,
                records["entity_groups"],
                """
                UNWIND $rows AS row
                MATCH (e:Entity {graph: row.graph, norm: row.entity_norm})
                MATCH (g:EntityGroup {graph: row.graph, group_key: row.group_key})
                CREATE (e)-[:IN_GROUP]->(g)
                """,
            )
            _run_write_batches(
                session,
                records["community_docs"],
                """
                UNWIND $rows AS row
                MATCH (c:Community {graph: row.graph, community_id: row.community_id})
                MATCH (d:Document {graph: row.graph, document_id: row.document_id})
                CREATE (d)-[:IN_COMMUNITY]->(c)
                """,
            )
            _run_write_batches(
                session,
                records["community_entities"],
                """
                UNWIND $rows AS row
                MATCH (c:Community {graph: row.graph, community_id: row.community_id})
                MATCH (e:Entity {graph: row.graph, norm: row.entity_norm})
                CREATE (c)-[:HAS_ENTITY]->(e)
                """,
            )
            _run_write_batches(
                session,
                records["related_docs"],
                """
                UNWIND $rows AS row
                MATCH (left:Document {graph: row.graph, document_id: row.left_document_id})
                MATCH (right:Document {graph: row.graph, document_id: row.right_document_id})
                CREATE (left)-[:RELATED_TO {weight: row.weight}]->(right)
                CREATE (right)-[:RELATED_TO {weight: row.weight}]->(left)
                """,
            )

    load_graph_snapshot.cache_clear()
    return {
        "documents": len(records["documents"]),
        "entities": len(records["entities"]),
        "communities": len(records["communities"]),
        "related_edges": len(records["related_docs"]),
    }


def graph_layer_stats(graph_name: str) -> dict[str, int]:
    with _driver() as driver:
        with driver.session() as session:
            document_count = int(
                session.run(
                    """
                    MATCH (d:Document {graph: $graph})
                    RETURN count(d) AS count
                    """,
                    graph=graph_name,
                ).single()["count"]
            )
            entity_count = int(
                session.run(
                    """
                    MATCH (e:Entity {graph: $graph})
                    RETURN count(e) AS count
                    """,
                    graph=graph_name,
                ).single()["count"]
            )
            community_count = int(
                session.run(
                    """
                    MATCH (c:Community {graph: $graph})
                    RETURN count(c) AS count
                    """,
                    graph=graph_name,
                ).single()["count"]
            )
            related_edges = int(
                session.run(
                    """
                    MATCH (:Document {graph: $graph})-[r:RELATED_TO]->(:Document {graph: $graph})
                    RETURN count(r) AS count
                    """,
                    graph=graph_name,
                ).single()["count"]
            )
    return {
        "documents": document_count,
        "entities": entity_count,
        "communities": community_count,
        "related_edges": related_edges,
    }


def clone_graph_layer(source_graph: str, target_graph: str, variant: str) -> dict[str, int]:
    if source_graph == target_graph:
        return graph_layer_stats(target_graph)

    with _driver() as driver:
        with driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE n.graph = $graph
                DETACH DELETE n
                """,
                graph=target_graph,
            )
            session.run(
                """
                MATCH (d:Document {graph: $source})
                CREATE (:Document {
                  graph: $target,
                  variant: $variant,
                  document_id: d.document_id,
                  title: d.title,
                  text: d.text,
                  entity_count: d.entity_count,
                  pagerank: d.pagerank
                })
                """,
                source=source_graph,
                target=target_graph,
                variant=variant,
            )
            session.run(
                """
                MATCH (e:Entity {graph: $source})
                CREATE (:Entity {
                  graph: $target,
                  variant: $variant,
                  norm: e.norm,
                  name: e.name,
                  frequency: e.frequency,
                  group_key: e.group_key
                })
                """,
                source=source_graph,
                target=target_graph,
                variant=variant,
            )
            session.run(
                """
                MATCH (g:EntityGroup {graph: $source})
                CREATE (:EntityGroup {
                  graph: $target,
                  variant: $variant,
                  group_key: g.group_key,
                  label: g.label
                })
                """,
                source=source_graph,
                target=target_graph,
                variant=variant,
            )
            session.run(
                """
                MATCH (c:Community {graph: $source})
                CREATE (:Community {
                  graph: $target,
                  variant: $variant,
                  community_id: replace(c.community_id, $source, $target),
                  summary: c.summary,
                  size: c.size
                })
                """,
                source=source_graph,
                target=target_graph,
                variant=variant,
            )
            session.run(
                """
                MATCH (d:Document {graph: $source})-[:MENTIONS]->(e:Entity {graph: $source})
                MATCH (d2:Document {graph: $target, document_id: d.document_id})
                MATCH (e2:Entity {graph: $target, norm: e.norm})
                CREATE (d2)-[:MENTIONS]->(e2)
                """,
                source=source_graph,
                target=target_graph,
            )
            session.run(
                """
                MATCH (e:Entity {graph: $source})-[:IN_GROUP]->(g:EntityGroup {graph: $source})
                MATCH (e2:Entity {graph: $target, norm: e.norm})
                MATCH (g2:EntityGroup {graph: $target, group_key: g.group_key})
                CREATE (e2)-[:IN_GROUP]->(g2)
                """,
                source=source_graph,
                target=target_graph,
            )
            session.run(
                """
                MATCH (d:Document {graph: $source})-[:IN_COMMUNITY]->(c:Community {graph: $source})
                MATCH (d2:Document {graph: $target, document_id: d.document_id})
                MATCH (c2:Community {
                  graph: $target,
                  community_id: replace(c.community_id, $source, $target)
                })
                CREATE (d2)-[:IN_COMMUNITY]->(c2)
                """,
                source=source_graph,
                target=target_graph,
            )
            session.run(
                """
                MATCH (c:Community {graph: $source})-[:HAS_ENTITY]->(e:Entity {graph: $source})
                MATCH (c2:Community {
                  graph: $target,
                  community_id: replace(c.community_id, $source, $target)
                })
                MATCH (e2:Entity {graph: $target, norm: e.norm})
                CREATE (c2)-[:HAS_ENTITY]->(e2)
                """,
                source=source_graph,
                target=target_graph,
            )
            session.run(
                """
                MATCH (left:Document {graph: $source})-[r:RELATED_TO]->(right:Document {graph: $source})
                MATCH (left2:Document {graph: $target, document_id: left.document_id})
                MATCH (right2:Document {graph: $target, document_id: right.document_id})
                CREATE (left2)-[:RELATED_TO {weight: r.weight}]->(right2)
                """,
                source=source_graph,
                target=target_graph,
            )

    load_graph_snapshot.cache_clear()
    return graph_layer_stats(target_graph)


@lru_cache(maxsize=16)
def load_graph_snapshot(graph_name: str) -> GraphSnapshot:
    documents: dict[str, dict[str, object]] = {}
    entities: dict[str, dict[str, object]] = {}
    communities: dict[str, dict[str, object]] = {}
    groups: dict[str, dict[str, object]] = {}
    doc_entities: dict[str, set[str]] = defaultdict(set)
    entity_docs: dict[str, set[str]] = defaultdict(set)
    community_docs: dict[str, set[str]] = defaultdict(set)
    community_entities: dict[str, set[str]] = defaultdict(set)
    group_entities: dict[str, set[str]] = defaultdict(set)
    graph = nx.Graph()

    with _driver() as driver:
        with driver.session() as session:
            for record in session.run(
                """
                MATCH (d:Document {graph: $graph})
                RETURN d.document_id AS document_id, d.title AS title, d.text AS text, d.pagerank AS pagerank
                """,
                graph=graph_name,
            ):
                documents[record["document_id"]] = {
                    "title": record["title"],
                    "text": record["text"],
                    "pagerank": float(record["pagerank"] or 0.0),
                }
                graph.add_node(record["document_id"], kind="document")

            for record in session.run(
                """
                MATCH (e:Entity {graph: $graph})
                RETURN e.norm AS norm, e.name AS name, e.group_key AS group_key, e.frequency AS frequency
                """,
                graph=graph_name,
            ):
                entities[record["norm"]] = {
                    "name": record["name"],
                    "group_key": record["group_key"],
                    "frequency": int(record["frequency"] or 0),
                }
                graph.add_node(record["norm"], kind="entity")

            for record in session.run(
                """
                MATCH (c:Community {graph: $graph})
                RETURN c.community_id AS community_id, c.summary AS summary, c.size AS size
                """,
                graph=graph_name,
            ):
                communities[record["community_id"]] = {
                    "summary": record["summary"],
                    "size": int(record["size"] or 0),
                }

            for record in session.run(
                """
                MATCH (g:EntityGroup {graph: $graph})
                RETURN g.group_key AS group_key, g.label AS label
                """,
                graph=graph_name,
            ):
                groups[record["group_key"]] = {"label": record["label"]}

            for record in session.run(
                """
                MATCH (d:Document {graph: $graph})-[:MENTIONS]->(e:Entity {graph: $graph})
                RETURN d.document_id AS document_id, e.norm AS norm
                """,
                graph=graph_name,
            ):
                doc_entities[record["document_id"]].add(record["norm"])
                entity_docs[record["norm"]].add(record["document_id"])
                graph.add_edge(record["document_id"], record["norm"], weight=1.0)

            for record in session.run(
                """
                MATCH (d:Document {graph: $graph})-[:IN_COMMUNITY]->(c:Community {graph: $graph})
                RETURN d.document_id AS document_id, c.community_id AS community_id
                """,
                graph=graph_name,
            ):
                community_docs[record["community_id"]].add(record["document_id"])

            for record in session.run(
                """
                MATCH (c:Community {graph: $graph})-[:HAS_ENTITY]->(e:Entity {graph: $graph})
                RETURN c.community_id AS community_id, e.norm AS norm
                """,
                graph=graph_name,
            ):
                community_entities[record["community_id"]].add(record["norm"])

            for record in session.run(
                """
                MATCH (e:Entity {graph: $graph})-[:IN_GROUP]->(g:EntityGroup {graph: $graph})
                RETURN e.norm AS norm, g.group_key AS group_key
                """,
                graph=graph_name,
            ):
                group_entities[record["group_key"]].add(record["norm"])

            for record in session.run(
                """
                MATCH (left:Document {graph: $graph})-[r:RELATED_TO]->(right:Document {graph: $graph})
                RETURN left.document_id AS left_document_id, right.document_id AS right_document_id, r.weight AS weight
                """,
                graph=graph_name,
            ):
                graph.add_edge(
                    record["left_document_id"],
                    record["right_document_id"],
                    weight=float(record["weight"] or 1.0),
                )

    return GraphSnapshot(
        graph_name=graph_name,
        documents=documents,
        entities=entities,
        communities=communities,
        groups=groups,
        doc_entities={key: set(value) for key, value in doc_entities.items()},
        entity_docs={key: set(value) for key, value in entity_docs.items()},
        community_docs={key: set(value) for key, value in community_docs.items()},
        community_entities={key: set(value) for key, value in community_entities.items()},
        group_entities={key: set(value) for key, value in group_entities.items()},
        graph=graph,
    )


def graph_query_terms(query: str) -> set[str]:
    return {token for token in _normalize_tokens(query) if len(token) >= 3}


def match_query_entities(snapshot: GraphSnapshot, query: str, limit: int = 6) -> list[str]:
    terms = graph_query_terms(query)
    if not terms:
        return []

    scored: list[tuple[int, int, str]] = []
    for norm, entity in snapshot.entities.items():
        entity_terms = set(norm.split())
        overlap = len(entity_terms & terms)
        if not overlap:
            continue
        scored.append((overlap, int(entity.get("frequency", 0)), norm))

    scored.sort(reverse=True)
    return [norm for _, _, norm in scored[:limit]]


def fetch_graph_document_texts(graph_name: str, document_ids: list[str]) -> dict[str, dict[str, str]]:
    if not document_ids:
        return {}
    with _driver() as driver:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (d:Document {graph: $graph})
                WHERE d.document_id IN $document_ids
                RETURN d.document_id AS document_id, d.title AS title, d.text AS text
                """,
                graph=graph_name,
                document_ids=document_ids,
            )
            return {
                row["document_id"]: {"title": row["title"], "text": row["text"]}
                for row in rows
            }
