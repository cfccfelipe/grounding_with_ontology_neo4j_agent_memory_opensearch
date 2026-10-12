import json
import time
import logging
import boto3
from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    AWS_REGION,
    BEDROCK_EMBEDDING_MODEL_ID,
    EMBEDDING_DIMENSIONS,
)

logger = logging.getLogger(__name__)


class GraphLabDriver:
    def __init__(self, connect_retries: int = 30, retry_delay: float = 2.0):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._wait_for_neo4j(connect_retries, retry_delay)
        self.bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    def _wait_for_neo4j(self, retries: int, delay: float) -> None:
        for attempt in range(1, retries + 1):
            try:
                self.driver.verify_connectivity()
                logger.info("Connected to Neo4j at %s", NEO4J_URI)
                return
            except Exception as exc:  # noqa: BLE001
                logger.info("Neo4j not ready (attempt %d/%d): %s", attempt, retries, exc)
                time.sleep(delay)
        raise RuntimeError(f"Neo4j did not become reachable at {NEO4J_URI}")

    def close(self):
        self.driver.close()

    def get_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding using AWS Bedrock Titan Text Embeddings V2.

        Titan V2 supports 256/512/1024 dimensions (default 1024). normalize=True
        returns unit vectors, so inner-product == cosine downstream.
        """
        payload = {
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }
        response = self.bedrock.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(response["body"].read().decode("utf-8"))
        return body["embedding"]
