import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD is generated per launch (Secrets Manager) and injected into
# the instance .env at boot; there is no shared literal default.
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SIGV4 = os.getenv("OPENSEARCH_USE_SIGV4", "false").lower() == "true"
OPENSEARCH_SERVICE = os.getenv("OPENSEARCH_SERVICE", "aoss")
OPENSEARCH_ENGINE = os.getenv("OPENSEARCH_ENGINE", "faiss")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
BEDROCK_LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "us.amazon.nova-pro-v1:0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# DynamoDB conversational-memory table (Agent A). PK session_id (S), SK turn_id (N).
SESSION_TABLE = os.getenv("SESSION_TABLE", "claims-sessions")

# Aurora PostgreSQL (pgvector) — Agent C vector store B + governed-metric source of record.
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "coa")
PGUSER = os.getenv("PGUSER", "coaadmin")
PGPASSWORD = os.getenv("PGPASSWORD", "")

# Evidence publishing (Model B): the learner produces DIY artifacts and uploads
# them to s3://$EVIDENCE_BUCKET/$EVIDENCE_PREFIX/... ; the grader reads them via
# boto3 (s3:GetObject). Set in the instance .env by the CDK.
EVIDENCE_BUCKET = os.getenv("EVIDENCE_BUCKET", "")
EVIDENCE_PREFIX = os.getenv("EVIDENCE_PREFIX", "evidence")
