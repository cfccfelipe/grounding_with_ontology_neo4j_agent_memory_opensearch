import json
import time
import logging

from src.graph_client import GraphLabDriver
from src.opensearch_client import OpenSearchVectorStore, INDEX_NAME
from src.pgvector_client import PgVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared query set: one keyword-leaning, one semantic-leaning, one governed.
QUERIES = [
    "exclusions for Nexus Tower",
    "commercial property catastrophic risk",
    "which solvency metric applies to commercial policies",
]


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[len(s) // 2], 2)


def run(out_path: str = "comparison_results.json") -> dict:
    lab = GraphLabDriver()
    os_store = OpenSearchVectorStore()
    os_store.wait_until_ready()
    pg = PgVectorStore()
    try:
        results: dict[str, dict] = {}

        for name in ("opensearch", "pgvector"):
            latencies: list[float] = []
            hits_total = 0
            for q in QUERIES:
                vec = lab.get_embedding(q)
                t0 = time.perf_counter()
                if name == "opensearch":
                    hits = os_store.hybrid_vector_search(q, vec, top_k=3)
                else:
                    hits = pg.knn(vec, top_k=3)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                hits_total += len(hits)

            doc_count = (
                os_store.client.count(index=INDEX_NAME)["count"]
                if name == "opensearch"
                else pg.count()
            )
            results[name] = {
                "doc_count": int(doc_count),
                "p50_ms": _p50(latencies),
                "relevance_at_k": round(hits_total / (len(QUERIES) * 3), 3),
            }

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s: %s", out_path, results)
        return results
    finally:
        lab.close()
        pg.close()


def preview(out_path: str = "sample_comparison.json") -> dict:
    """Practice-safe like-for-like preview of the two vector stores."""
    lab = GraphLabDriver()
    os_store = OpenSearchVectorStore()
    os_store.wait_until_ready()
    pg = PgVectorStore()
    try:
        vec = lab.get_embedding("commercial property catastrophic risk")
        results = {
            "opensearch": {
                "doc_count": int(os_store.client.count(index=INDEX_NAME)["count"]),
                "knn_hits": len(os_store._knn(vec, 3)),
            },
            "pgvector": {
                "doc_count": pg.count(),
                "knn_hits": len(pg.knn(vec, 3)),
            },
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s: %s", out_path, results)
        return results
    finally:
        lab.close()
        pg.close()


if __name__ == "__main__":
    import sys

    if "--preview" in sys.argv:
        print(json.dumps(preview(), indent=2))
    else:
        print(json.dumps(run(), indent=2))
