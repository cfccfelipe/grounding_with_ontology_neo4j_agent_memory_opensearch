import json
import sys

from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore
from src.pgvector_client import PgVectorStore
from src.retriever import SemanticGraphRetriever

ONTOLOGY_OUT = "ontology_result.json"
RETRIEVAL_OUT = "retrieval_result.json"

# The two lines the DIY-2 traversal must resolve (existing + learner-authored).
_ONTOLOGY_QUERY = """
UNWIND ['ent_nexus_tower', 'ent_atlantic_freight'] AS ent_id
MATCH (e:Entity {id: ent_id})
OPTIONAL MATCH (e)-[:INSTANCE_OF]->(c:OntologyClass)<-[:COMPUTED_OVER]-(m:GovernedMetric)
RETURN e.id AS entity_id,
       c.name AS class_name,
       collect(DISTINCT m.name) AS applicable_metrics
ORDER BY entity_id
"""


def emit_ontology() -> dict:
    """DIY-2 evidence: 3-hop traversal result for both insurance lines."""
    lab = GraphLabDriver()
    try:
        with lab.driver.session(default_access_mode="r") as session:
            records = [r.data() for r in session.run(_ONTOLOGY_QUERY)]
    finally:
        lab.close()
    result = {"evidence": "ontology", "traversal": records}
    with open(ONTOLOGY_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {ONTOLOGY_OUT}:")
    print(json.dumps(result, indent=2))
    print(
        f"\nNext: aws s3 cp {ONTOLOGY_OUT} "
        f"s3://$EVIDENCE_BUCKET/evidence/{ONTOLOGY_OUT}"
    )
    return result


def emit_retrieval() -> dict:
    """DIY-3 evidence: RRF hybrid-retrieval probe (each hit must carry a score)."""
    lab = GraphLabDriver()
    os_store = OpenSearchVectorStore()
    os_store.wait_until_ready()
    try:
        retriever = SemanticGraphRetriever(lab, os_store)
        hits = retriever.hybrid_rag_traversal(
            "Nexus Tower commercial property exclusions", top_k=2
        )
        probe = [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]
    finally:
        lab.close()
    result = {"evidence": "retrieval", "rrf_probe": probe}
    with open(RETRIEVAL_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {RETRIEVAL_OUT}:")
    print(json.dumps(result, indent=2))
    print(
        f"\nNext: aws s3 cp {RETRIEVAL_OUT} "
        f"s3://$EVIDENCE_BUCKET/evidence/{RETRIEVAL_OUT}"
    )
    return result


def preview() -> dict:
    """Practice-safe preview of the evidence workflow, using REAL live state.
    This does NOT run the real ``ontology`` or ``retrieval`` evidence.
    """
    from src.opensearch_client import INDEX_NAME

    # The INSTANCE_OF relationship does not exist until DIY-2, so counting it
    # emits a harmless "unknown relationship type" notification. Silence the
    # neo4j notification logger for this preview so the output stays clean.
    import logging as _logging
    _logging.getLogger("neo4j.notifications").setLevel(_logging.ERROR)

    lab = GraphLabDriver()
    os_store = OpenSearchVectorStore()
    os_store.wait_until_ready()
    pg = PgVectorStore()
    try:
        graph_counts_query = """
        RETURN
          COUNT { (c:OntologyClass) } AS ontology_classes,
          COUNT { (m:GovernedMetric) } AS governed_metrics,
          COUNT { (e:Entity) } AS entities,
          COUNT { (:Entity)-[:INSTANCE_OF]->(:OntologyClass) } AS instance_of_links
        """
        with lab.driver.session(default_access_mode="r") as session:
            graph = session.run(graph_counts_query).single().data()

        sample = {
            "vector_stores": {
                "opensearch": {"doc_count": int(os_store.client.count(index=INDEX_NAME)["count"])},
                "pgvector": {"doc_count": pg.count()},
            },
            "graph": graph,
        }
    finally:
        lab.close()
        pg.close()

    with open("sample_evidence.json", "w") as f:
        json.dump(sample, f, indent=2)
    print("Wrote sample_evidence.json")
    print(json.dumps(sample, indent=2))
    return sample


def emit_cypher_guard(out_path: str = "cypher_guard_result.json") -> dict:
    """DIY-4 evidence: exercise the read-Cypher guard with fixed probes.

    Runs one allowed read and one disallowed query through the governed
    `execute_read_cypher` MCP tool and records whether each was handled
    correctly. The probe queries are fixed here (not learner-supplied), so the
    flags reflect the learner's guard implementation, not their choice of query.
    A response beginning with 'Rejected:' means the guard refused the query.
    """
    from src import mcp_server

    allowed_probe = "MATCH (c:OntologyClass) RETURN c.name"
    blocked_probe = "CALL dbms.components() YIELD name RETURN name"

    allowed_resp = mcp_server.execute_read_cypher(allowed_probe)
    blocked_resp = mcp_server.execute_read_cypher(blocked_probe)

    result = {
        "evidence": "cypher-guard",
        "allowed_query_ok": not allowed_resp.startswith("Rejected:"),
        "blocked_query_rejected": blocked_resp.startswith("Rejected:"),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}:")
    print(json.dumps(result, indent=2))
    return result


def emit_mcp_tools(out_path: str = "mcp_tools_result.json") -> dict:
    """DIY-1 evidence: confirm the two memory MCP tools are registered AND that
    ``get_session`` returns turns oldest-first.

    Imports the governed MCP server and lists its registered tools, recording
    whether `append_turn` and `get_session` are present (the learner's
    tool-registration work at the `# TODO(DIY-1)` marker).

    It also calls the learner's ``get_session('s-demo')`` on the box (against the
    real DynamoDB table) and records the returned ``turn_id``s in the EXACT order
    returned -- deliberately UN-sorted -- so the grader can assert the ordering
    fix (``ScanIndexForward=True`` in ``memory_client.py``) without importing
    on-box code. The seeded demo session has turns 1, 2, 3, so the correct
    chronological order is ``[1, 2, 3]``; the reversed defect yields ``[3, 2, 1]``.
    """
    import asyncio
    from src import mcp_server

    tool_names = sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools()))

    # Capture the ACTUAL order get_session returns -- do NOT sort. This reflects
    # the learner's ScanIndexForward fix. A cold/unseeded session yields [].
    session_order = [t["turn_id"] for t in mcp_server.memory.get_session("s-demo")]

    result = {
        "evidence": "mcp-tools",
        "append_turn_registered": "append_turn" in tool_names,
        "get_session_registered": "get_session" in tool_names,
        "tools": tool_names,
        "session_order": session_order,
        "session_order_correct": session_order == [1, 2, 3],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}:")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "ontology":
        emit_ontology()
    elif which == "retrieval":
        emit_retrieval()
    elif which == "preview":
        preview()
    elif which == "cypher-guard":
        emit_cypher_guard()
    elif which == "mcp-tools":
        emit_mcp_tools()
    else:
        print("usage: python -m src.emit_evidence [ontology|retrieval|preview|cypher-guard|mcp-tools]")
        sys.exit(2)


if __name__ == "__main__":
    main()
