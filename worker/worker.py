"""interaction-worker: consume la cola 'interactions.persist' de RabbitMQ y guarda cada evento
en PostgreSQL de forma idempotente (event_id como clave primaria). Si un mensaje falla 3 veces
se rechaza sin reencolar y RabbitMQ lo envía a la dead-letter queue 'interactions.dlq'."""
import json
import logging
import os
import time

import pika
import psycopg

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://shopfast:shopfast@postgres:5432/shopfast")
EXCHANGE, QUEUE, DLX, DLQ = "interactions", "interactions.persist", "interactions.dlx", "interactions.dlq"
VALID_TYPES = {"view", "cart", "purchase"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("worker")


def setup(ch):
    ch.exchange_declare(EXCHANGE, exchange_type="fanout", durable=True)
    ch.exchange_declare(DLX, exchange_type="fanout", durable=True)
    ch.queue_declare(DLQ, durable=True)
    ch.queue_bind(DLQ, DLX)
    ch.queue_declare(QUEUE, durable=True, arguments={"x-dead-letter-exchange": DLX})
    ch.queue_bind(QUEUE, EXCHANGE)
    ch.basic_qos(prefetch_count=20)


def validate(ev: dict) -> None:
    if ev.get("event_type") not in VALID_TYPES or not ev.get("event_id"):
        raise ValueError("evento inválido")
    int(ev["user_id"]), int(ev["product_id"])


def main():
    db = None
    while db is None:
        try:
            db = psycopg.connect(DATABASE_URL, autocommit=True)
        except psycopg.OperationalError:
            time.sleep(2)
    while True:
        try:
            conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            break
        except pika.exceptions.AMQPConnectionError:
            time.sleep(2)
    ch = conn.channel()
    setup(ch)

    def on_message(ch, method, props, body):
        try:
            ev = json.loads(body)
            validate(ev)
            db.execute("INSERT INTO interactions (event_id, user_id, product_id, event_type, occurred_at) "
                       "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING",
                       (ev["event_id"], int(ev["user_id"]), int(ev["product_id"]), ev["event_type"], ev["occurred_at"]))
            ch.basic_ack(method.delivery_tag)
            log.info(json.dumps({"event": "persisted", "event_id": ev["event_id"], "type": ev["event_type"]}))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            ch.basic_reject(method.delivery_tag, requeue=False)          # → DLQ
            log.info(json.dumps({"event": "rejected_to_dlq", "error": str(exc)}))

    ch.basic_consume(QUEUE, on_message)
    log.info(json.dumps({"event": "worker_started", "queue": QUEUE}))
    ch.start_consuming()


if __name__ == "__main__":
    main()
