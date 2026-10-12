import json
import logging

import boto3

from src.config import AWS_REGION, BEDROCK_LLM_MODEL_ID
from src.graph_client import GraphLabDriver
from src.memory_client import SessionMemoryStore
from src.metric_service import compute_governed_metric
from src.opensearch_client import OpenSearchVectorStore
from src.pgvector_client import PgVectorStore
from src.retriever import SemanticGraphRetriever

logger = logging.getLogger(__name__)

REFUSAL = (
    "I cannot ground an answer: no verified metric or document context was retrieved."
)

# Maps an ontology class name to the relational line_of_business key used by the
# governed-metric source of record (Aurora).
LOB_BY_CLASS = {
    "CommercialPropertyPolicy": "commercial_property",
    "MarineCargoPolicy": "marine_cargo",
}


class GraphGroundedAssistant:
    """Bedrock Converse grounded on DynamoDB memory + Neo4j graph + OpenSearch RAG,
    with governed metrics resolved from the graph and computed over Aurora.

    Answers strictly from retrieved context; refuses when nothing is retrieved.
    """

    def __init__(self):
        self.lab = GraphLabDriver()
        self.os_store = OpenSearchVectorStore()
        self.os_store.wait_until_ready()
        self.retriever = SemanticGraphRetriever(self.lab, self.os_store)
        self.memory = SessionMemoryStore()
        self.pg = PgVectorStore()
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    def invoke(self, session_id: str, user_prompt: str) -> str:
        history = self.memory.get_session(session_id)
        metrics = self.retriever.resolve_governed_metric(
            user_prompt, top_k=1, threshold=0.75
        )
        rag_context = self.retriever.hybrid_rag_traversal(user_prompt, top_k=2)

        # Hallucination gate: refuse when there is nothing to ground on.
        if not metrics and not rag_context:
            return REFUSAL

        # Compute each resolved governed metric over the Aurora source of record.
        computed = []
        for m in metrics:
            lob = None
            for cls in m.get("scoped_classes", []):
                if cls in LOB_BY_CLASS:
                    lob = LOB_BY_CLASS[cls]
                    break
            if lob:
                computed.append(
                    compute_governed_metric(self.pg, m["metric_name"], lob)
                )

        system_prompt = (
            "You are an enterprise AI analyst grounded on DynamoDB (memory), "
            "Neo4j (semantic graph), OpenSearch (document RAG), and AWS Bedrock.\n"
            "Answer STRICTLY from the verified context below. "
            "If the context is empty, say you cannot ground an answer.\n\n"
            f"CONVERSATION HISTORY: {json.dumps(history)}\n"
            f"GOVERNED METRICS (definitions): {json.dumps(metrics)}\n"
            f"COMPUTED VALUES (source of record): {json.dumps(computed)}\n"
            f"GRAPH RAG CONTEXT: {json.dumps(rag_context)}\n"
        )

        response = self.bedrock_runtime.converse(
            modelId=BEDROCK_LLM_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
        )
        return response["output"]["message"]["content"][0]["text"]

    def close(self):
        self.lab.close()
        self.pg.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = GraphGroundedAssistant()
    try:
        print("\n--- Bedrock Converse (graph-grounded) ---")
        answer = agent.invoke(
            "s-demo", "What are the insurance exclusions for Nexus Tower?"
        )
        print(f"\nModel Response:\n{answer}")
    finally:
        agent.close()
