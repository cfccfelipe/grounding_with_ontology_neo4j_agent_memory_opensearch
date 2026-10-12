import logging
from typing import Any
from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore

logger = logging.getLogger(__name__)


class SemanticGraphRetriever:
    def __init__(self, lab: GraphLabDriver, os_store: OpenSearchVectorStore):
        self.lab = lab
        self.os_store = os_store

    def resolve_governed_metric(self, query_text, top_k=1, threshold=0.80) -> list[dict[str, Any]]:
        query_vector = self.lab.get_embedding(query_text)
        cypher = """
        CALL db.index.vector.queryNodes('governed_metric_vector_idx', $top_k, $query_vector)
        YIELD node AS metric, score
        WHERE score >= $threshold
        MATCH (metric)-[:COMPUTED_OVER]->(c:OntologyClass)
        RETURN metric.name AS metric_name,
               metric.description AS description,
               metric.calculation_template AS template,
               metric.properties_used AS properties,
               collect(c.name) AS scoped_classes,
               score
        ORDER BY score DESC
        """
        with self.lab.driver.session() as session:
            result = session.run(cypher, top_k=top_k, query_vector=query_vector, threshold=threshold)
            return [record.data() for record in result]

    def hybrid_rag_traversal(self, query_text, top_k=3) -> list[dict[str, Any]]:
        query_vector = self.lab.get_embedding(query_text)
        os_results = self.os_store.hybrid_vector_search(query_text, query_vector, top_k=top_k)
        if not os_results:
            return []

        all_entity_ids = list({e for res in os_results for e in res.get("entity_ids", [])})

        cypher = """
        UNWIND $entity_ids AS ent_id
        MATCH (e:Entity {id: ent_id})
        OPTIONAL MATCH (e)-[:INSTANCE_OF]->(c:OntologyClass)
        OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
        OPTIONAL MATCH (m:GovernedMetric)-[:COMPUTED_OVER]->(c)
        RETURN e.id AS entity_id,
               e.name AS entity_name,
               c.name AS class_name,
               collect(DISTINCT {relation: type(r), neighbor: neighbor.name}) AS graph_connections,
               collect(DISTINCT m.name) AS applicable_metrics
        """
        with self.lab.driver.session() as session:
            graph_records = session.run(cypher, entity_ids=all_entity_ids).data() if all_entity_ids else []

        graph_lookup = {rec["entity_id"]: rec for rec in graph_records}

        fused = []
        for hit in os_results:
            # Extract score using fallback key resolution
            score = float(hit.get("score") or hit.get("fused_score") or hit.get("_score") or 0.0)

            fused.append({
                "chunk_id": hit["chunk_id"],
                "chunk_text": hit.get("text") or hit.get("chunk_text") or "",
                "source": hit.get("source_uri") or hit.get("source") or "",
                "score": score,
                "grounded_entities": [
                    graph_lookup[eid] for eid in hit.get("entity_ids", []) if eid in graph_lookup
                ],
            })
        return fused