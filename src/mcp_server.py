import json
import logging
from mcp.server.fastmcp import FastMCP
from neo4j.exceptions import ClientError
from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore
from src.retriever import SemanticGraphRetriever
from src.memory_client import SessionMemoryStore
from src.pgvector_client import PgVectorStore
from src.metric_service import compute_governed_metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("Neo4j-COA-KnowledgeGraph")

lab = GraphLabDriver()
os_store = OpenSearchVectorStore()
os_store.wait_until_ready()
retriever = SemanticGraphRetriever(lab, os_store)
memory = SessionMemoryStore()

# Disallowed tokens for the governed read-Cypher tool: procedure calls, admin
# namespaces, bulk load, and any write clause. Matched case-insensitively.
_FORBIDDEN_CYPHER = (
    "CALL", "APOC", "DBMS.", "DB.", "LOAD CSV",
    "CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP",
)
pg_store = PgVectorStore()


@mcp.resource("ontology://schema")
def get_ontology_schema() -> str:
    cypher = """
    MATCH (c:OntologyClass)
    OPTIONAL MATCH (c)<-[:SUBCLASS_OF]-(sub:OntologyClass)
    OPTIONAL MATCH (m:GovernedMetric)-[:COMPUTED_OVER]->(c)
    RETURN c.name AS class_name,
           c.description AS description,
           collect(DISTINCT sub.name) AS subclasses,
           collect(DISTINCT m.name) AS associated_metrics
    """
    with lab.driver.session(default_access_mode="r") as session:
        records = [r.data() for r in session.run(cypher)]
    return json.dumps(records, indent=2)


@mcp.tool()
def resolve_governed_metric(business_question: str) -> str:
    results = retriever.resolve_governed_metric(business_question, top_k=2, threshold=0.75)
    return json.dumps(results or "No pre-compiled governed metric found.", indent=2)


@mcp.tool()
def compute_metric(metric_name: str, line_of_business: str) -> str:
    """Compute an approved governed metric against the Aurora source of record
    (parameterized, validated, read-only). Returns the real value, e.g. 1.85."""
    return json.dumps(compute_governed_metric(pg_store, metric_name, line_of_business))


# TODO(DIY-1): register the Memory MCP tools here.
# ---------------------------------------------------------------------------
# DIY-1 Memory MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_session(session_id: str) -> str:
    """Retrieve conversational turns for a session in chronological order."""
    turns = memory.get_session(session_id)
    return json.dumps(turns, indent=2)


@mcp.tool()
def append_turn(session_id: str, role: str, content: str) -> str:
    """Append a new conversational turn (e.g., 'user' or 'assistant') to session memory."""
    turn_id = memory.append_turn(session_id, role, content)
    return json.dumps({"status": "success", "turn_id": turn_id}, indent=2)


@mcp.tool()
def hybrid_graph_search(query: str, top_k: int = 3) -> str:
    results = retriever.hybrid_rag_traversal(query, top_k=top_k)
    return json.dumps(results or "No context found.", indent=2)


@mcp.tool()
def execute_read_cypher(cypher_query: str) -> str:
    upper_query = cypher_query.upper()

    # 1. Reject forbidden keywords from the denylist
    if any(forbidden in upper_query for forbidden in _FORBIDDEN_CYPHER):
        return f"Rejected: Query contains forbidden clause or procedure."

    # 2. Reject queries that do not start with standard read clauses
    if not (upper_query.strip().startswith("MATCH") or upper_query.strip().startswith("OPTIONAL MATCH") or upper_query.strip().startswith("WITH") or upper_query.strip().startswith("UNWIND")):
        return f"Rejected: Query must begin with a read clause."

    def _read_tx(tx, query):
        return [record.data() for record in tx.run(query)]

    try:
        with lab.driver.session(default_access_mode="r") as session:
            records = session.execute_read(_read_tx, cypher_query)
            return json.dumps(records[:50], indent=2)
    except ClientError as ce:
        return f"Database Security/Syntax Error: {ce.message}"
    except Exception as e:  # noqa: BLE001
        return f"Execution Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
