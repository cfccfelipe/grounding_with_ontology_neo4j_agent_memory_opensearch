import logging

import psycopg2

from src.config import (
    AWS_REGION,
    EMBEDDING_DIMENSIONS,
    PGHOST,
    PGPORT,
    PGDATABASE,
    PGUSER,
    PGPASSWORD,
)

logger = logging.getLogger(__name__)


class PgVectorStore:
    """Aurora PostgreSQL (pgvector) — Agent C vector store B and the
    governed-metric source of record.

    Two tables:
      - ``coa_chunks``       — chunk vectors for the OpenSearch-vs-pgvector benchmark
      - ``quarterly_capital`` — the governed-metric source of record (DIY-4 compute)
    """

    def __init__(self, host: str = PGHOST, port: int = PGPORT,
                 dbname: str = PGDATABASE, user: str = PGUSER,
                 password: str = PGPASSWORD, region: str = AWS_REGION):
        self.host, self.port, self.dbname, self.region = host, port, dbname, region
        self.user, self.password = user, password
        self.conn = self._connect()
        self.conn.autocommit = True

    def _connect(self):
        if not self.password:
            raise RuntimeError(
                "No PGPASSWORD in env; cannot obtain Aurora credentials."
            )
        return psycopg2.connect(
            host=self.host, port=self.port, dbname=self.dbname,
            user=self.user, password=self.password, sslmode="require",
        )

    # --- schema ----------------------------------------------------------
    def initialize(self):
        """Idempotently create the pgvector extension, both tables, and the
        vector index. Safe to re-run (all IF NOT EXISTS)."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS coa_chunks (
                    chunk_id   text PRIMARY KEY,
                    text       text,
                    source_uri text,
                    embedding  vector({EMBEDDING_DIMENSIONS})
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS coa_chunks_embed_idx
                ON coa_chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS quarterly_capital (
                    line_of_business   text PRIMARY KEY,
                    eligible_own_funds numeric NOT NULL,
                    scr                numeric NOT NULL,
                    as_of              date    NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cargo_experience (
                    line_of_business text PRIMARY KEY,
                    incurred_losses  numeric NOT NULL,
                    earned_premium   numeric NOT NULL,
                    as_of            date    NOT NULL
                );
                """
            )
        logger.info("pgvector schema initialized (coa_chunks, quarterly_capital, cargo_experience).")

    def seed_quarterly_capital(self):
        """Seed the deterministic governed-metric source of record.

        commercial_property: 1_850_000 / 1_000_000 = 1.85 (asserted by DIY-4).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quarterly_capital (line_of_business, eligible_own_funds, scr, as_of)
                VALUES ('commercial_property', 1850000, 1000000, DATE '2026-06-30')
                ON CONFLICT (line_of_business) DO UPDATE
                  SET eligible_own_funds = EXCLUDED.eligible_own_funds,
                      scr = EXCLUDED.scr, as_of = EXCLUDED.as_of;
                """
            )
            # marine_cargo: 620_000 / 1_000_000 = 0.62 (the DIY-3-authored line)
            cur.execute(
                """
                INSERT INTO cargo_experience (line_of_business, incurred_losses, earned_premium, as_of)
                VALUES ('marine_cargo', 620000, 1000000, DATE '2026-06-30')
                ON CONFLICT (line_of_business) DO UPDATE
                  SET incurred_losses = EXCLUDED.incurred_losses,
                      earned_premium = EXCLUDED.earned_premium, as_of = EXCLUDED.as_of;
                """
            )
        logger.info("Seeded quarterly_capital (1.85) and cargo_experience (0.62).")

    # --- vector store B --------------------------------------------------
    def index_chunk(self, chunk_id, text, source_uri, embedding):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO coa_chunks (chunk_id, text, source_uri, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                  SET text = EXCLUDED.text,
                      source_uri = EXCLUDED.source_uri,
                      embedding = EXCLUDED.embedding;
                """,
                (chunk_id, text, source_uri, embedding),
            )

    def knn(self, query_vector, top_k: int = 3) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, source_uri,
                       1 - (embedding <=> %s::vector) AS score
                FROM coa_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, query_vector, top_k),
            )
            return [
                {"chunk_id": r[0], "text": r[1], "source_uri": r[2], "score": float(r[3])}
                for r in cur.fetchall()
            ]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM coa_chunks;")
            return int(cur.fetchone()[0])

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pg = PgVectorStore()
    try:
        pg.initialize()
        pg.seed_quarterly_capital()
        print("pgvector initialized; coa_chunks count:", pg.count())
    finally:
        pg.close()
