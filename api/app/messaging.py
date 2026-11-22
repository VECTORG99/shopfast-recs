"""Publicación de interacciones en RabbitMQ (exchange 'interactions', tipo fanout) y
lectura de interacciones recientes desde PostgreSQL (escritas por el worker)."""
from __future__ import annotations

import json
import os
import threading

import pika

EXCHANGE = "interactions"
QUEUE = "interactions.persist"


class Publisher:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
        self._conn = self._ch = None
        self._lock = threading.Lock()

    def _channel(self):
        if self._ch is None or self._ch.is_closed:
            self._conn = pika.BlockingConnection(pika.URLParameters(self.url))
            self._ch = self._conn.channel()
            self._ch.exchange_declare(EXCHANGE, exchange_type="fanout", durable=True)
            self._ch.queue_declare(QUEUE, durable=True, arguments={"x-dead-letter-exchange": "interactions.dlx"})
            self._ch.queue_bind(QUEUE, EXCHANGE)
            self._ch.confirm_delivery()
        return self._ch

    def publish(self, event: dict) -> None:
        with self._lock:
            try:
                self._channel().basic_publish(EXCHANGE, "", json.dumps(event).encode(),
                                              pika.BasicProperties(delivery_mode=2, content_type="application/json",
                                                                   message_id=event["event_id"]))
            except pika.exceptions.AMQPError:
                self._ch = None             # reconecta una vez
                self._channel().basic_publish(EXCHANGE, "", json.dumps(event).encode(),
                                              pika.BasicProperties(delivery_mode=2, message_id=event["event_id"]))


class InteractionStore:
    """Consulta en Postgres las interacciones recientes del usuario (las procesó el worker).
    Reutiliza una conexión y aplica un circuit breaker: si la BD falla, se omite 30 s y la
    API responde solo con el historial (degradación elegante, sin sumar latencia)."""

    def __init__(self, dsn: str | None = None, cooldown: float = 30.0):
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://shopfast:shopfast@postgres:5432/shopfast")
        self.cooldown, self._open_until, self._db = cooldown, 0.0, None
        self._lock = threading.Lock()

    def _conn(self):
        import psycopg
        if self._db is None or self._db.closed:
            self._db = psycopg.connect(self.dsn, connect_timeout=2, autocommit=True)
        return self._db

    def _query(self, sql, params=()):
        import time
        if time.monotonic() < self._open_until:
            raise ConnectionError("circuit breaker abierto")
        with self._lock:
            try:
                return self._conn().execute(sql, params).fetchall()
            except Exception:
                self._db, self._open_until = None, time.monotonic() + self.cooldown
                raise

    def live_counts(self, user_id: int) -> dict[int, dict]:
        try:
            rows = self._query("SELECT product_id, event_type, count(*) FROM interactions WHERE user_id=%s "
                               "GROUP BY product_id, event_type", (user_id,))
        except Exception:
            return {}
        out: dict[int, dict] = {}
        for pid, et, n in rows:
            out.setdefault(int(pid), {})[et] = int(n)
        return out

    def stats(self) -> dict:
        total = self._query("SELECT count(*) FROM interactions")[0][0]
        by_type = dict(self._query("SELECT event_type, count(*) FROM interactions GROUP BY 1"))
        recent = self._query("SELECT event_id, user_id, product_id, event_type, received_at FROM interactions "
                             "ORDER BY received_at DESC LIMIT 10")
        return {"total": total, "by_type": by_type,
                "recent": [{"event_id": str(r[0]), "user_id": str(r[1]), "product_id": r[2], "event_type": r[3],
                            "received_at": r[4].isoformat()} for r in recent]}
