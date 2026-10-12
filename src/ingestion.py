import logging
from typing import Any
from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_ontology_class(lab, uri, name, description, parent_uri=None):
    text_for_embedding = f"Class: {name}. Description: {description}"
    embedding = lab.get_embedding(text_for_embedding)
    cypher = """
    MERGE (c:OntologyClass {uri: $uri})
    SET c.name = $name,
        c.description = $description,
        c.embedding = $embedding,
        c.updated_at = timestamp()
    WITH c
    WHERE $parent_uri IS NOT NULL
    MERGE (p:OntologyClass {uri: $parent_uri})
    MERGE (c)-[:SUBCLASS_OF]->(p)
    """
    with lab.driver.session() as session:
        session.run(cypher, uri=uri, name=name, description=description,
                    embedding=embedding, parent_uri=parent_uri)
    logger.info("Ingested Ontology Class: %s", name)


def ingest_governed_metric(lab, name, description, calculation_template,
                           target_class_uris, properties_used):
    text_for_embedding = f"Metric: {name}. Description: {description}. Query: {calculation_template}"
    embedding = lab.get_embedding(text_for_embedding)
    cypher = """
    MERGE (m:GovernedMetric {name: $name})
    SET m.description = $description,
        m.calculation_template = $calculation_template,
        m.properties_used = $properties_used,
        m.embedding = $embedding,
        m.updated_at = timestamp()
    WITH m
    UNWIND $target_class_uris AS class_uri
    MATCH (c:OntologyClass {uri: class_uri})
    MERGE (m)-[:COMPUTED_OVER]->(c)
    """
    with lab.driver.session() as session:
        session.run(cypher, name=name, description=description,
                    calculation_template=calculation_template,
                    properties_used=properties_used, embedding=embedding,
                    target_class_uris=target_class_uris)
    logger.info("Ingested Governed Metric: %s", name)


def dual_ingest_document_chunk(lab, os_store, chunk_id, text, source_uri,
                               extracted_entities: list[dict[str, Any]], pg_store=None):
    # Chunk vector lives in OpenSearch (and optionally Aurora pgvector for the
    # benchmark); Neo4j stores structure only.
    embedding = lab.get_embedding(text)
    entity_ids = [e["id"] for e in extracted_entities]

    os_store.index_chunk(chunk_id=chunk_id, text=text, source_uri=source_uri,
                         entity_ids=entity_ids, embedding=embedding)

    if pg_store is not None:
        pg_store.index_chunk(chunk_id=chunk_id, text=text,
                             source_uri=source_uri, embedding=embedding)

    cypher = """
    MERGE (k:Chunk {id: $chunk_id})
    SET k.text = $text,
        k.source_uri = $source_uri,
        k.updated_at = timestamp()
    WITH k
    UNWIND $entities AS ent
    MERGE (e:Entity {id: ent.id})
    ON CREATE SET e.name = ent.name, e.type = ent.type
    MERGE (k)-[:MENTIONS {confidence: ent.confidence}]->(e)
    WITH e, ent
    WHERE ent.class_uri IS NOT NULL
    MERGE (c:OntologyClass {uri: ent.class_uri})
    MERGE (e)-[:INSTANCE_OF]->(c)
    """
    with lab.driver.session() as session:
        session.run(cypher, chunk_id=chunk_id, text=text, source_uri=source_uri,
                    entities=extracted_entities)
    logger.info("Dual-ingested chunk %s to OpenSearch & Neo4j.", chunk_id)
