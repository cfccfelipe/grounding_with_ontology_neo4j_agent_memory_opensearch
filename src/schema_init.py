import logging
from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore
from src.config import EMBEDDING_DIMENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INITIALIZATION_QUERIES = [
    "CREATE CONSTRAINT unique_class_uri IF NOT EXISTS FOR (c:OntologyClass) REQUIRE c.uri IS UNIQUE",
    "CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT unique_metric_name IF NOT EXISTS FOR (m:GovernedMetric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (k:Chunk) REQUIRE k.id IS UNIQUE",
    # Only the governed-metric vector index is queried (retriever.resolve_governed_metric).
    # Document-chunk vectors live in OpenSearch, so there is no Chunk vector index here.
    f"""
    CREATE VECTOR INDEX governed_metric_vector_idx IF NOT EXISTS
    FOR (m:GovernedMetric) ON (m.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIMENSIONS},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """,
]


def initialize_database(lab: GraphLabDriver, os_store: OpenSearchVectorStore):
    with lab.driver.session() as session:
        for query in INITIALIZATION_QUERIES:
            clean_q = " ".join(query.split())
            logger.info("Executing Cypher schema: %s", clean_q)
            session.run(query)

    os_store.initialize_index()
    logger.info("Database schemas, indexes, and constraints fully initialized.")


if __name__ == "__main__":
    lab = GraphLabDriver()
    os_store = OpenSearchVectorStore()
    os_store.wait_until_ready()
    try:
        initialize_database(lab, os_store)
    finally:
        lab.close()
