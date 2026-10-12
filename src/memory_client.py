import time
import logging

import boto3
from boto3.dynamodb.conditions import Key

from src.config import AWS_REGION, SESSION_TABLE

logger = logging.getLogger(__name__)


class SessionMemoryStore:
    """DynamoDB-backed conversational memory for Agent A (Intake).

    Contract (table ``claims-sessions``):
      - PK  ``session_id`` (S)
      - SK  ``turn_id``    (N)   monotonically increasing per session
      - attrs ``role`` (S), ``content`` (S), ``ts`` (N, epoch seconds)

    ``append_turn`` = PutItem;  ``get_session`` = Query by ``session_id``
    in ascending ``turn_id`` order. A cold/unknown session returns ``[]``.
    """

    def __init__(self, table_name: str = SESSION_TABLE, region: str = AWS_REGION):
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def append_turn(self, session_id: str, role: str, content: str) -> int:
        """Append one conversational turn; returns the assigned ``turn_id``."""
        existing = self.get_session(session_id)
        next_turn = (existing[-1]["turn_id"] + 1) if existing else 1
        self.table.put_item(
            Item={
                "session_id": session_id,
                "turn_id": next_turn,
                "role": role,
                "content": content,
                "ts": int(time.time()),
            }
        )
        logger.info("append_turn session=%s turn=%d role=%s", session_id, next_turn, role)
        return next_turn

    def seed_turn(self, session_id: str, turn_id: int, role: str, content: str) -> None:
        """Idempotently write one turn with an EXPLICIT ``turn_id``.

        Deploy-time seeding uses this instead of ``append_turn`` so the demo
        session is created correctly regardless of any learner-fixable defect in
        ``append_turn``/``get_session`` (which compute the next id by reading the
        session back). Direct ``PutItem`` with a fixed id is defect-independent.
        """
        self.table.put_item(
            Item={
                "session_id": session_id,
                "turn_id": turn_id,
                "role": role,
                "content": content,
                "ts": int(time.time()),
            }
        )
        logger.info("seed_turn session=%s turn=%d role=%s", session_id, turn_id, role)

    def get_session(self, session_id: str) -> list[dict]:
        """Return all turns for a session in ascending ``turn_id`` order.

        Returns ``[]`` for a cold/unknown session (never raises / never assumes
        a non-empty result).
        """
        resp = self.table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        items = resp.get("Items", [])
        return [
            {
                "turn_id": int(i["turn_id"]),
                "role": i["role"],
                "content": i["content"],
            }
            for i in items
        ]
