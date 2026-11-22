CREATE TABLE IF NOT EXISTS interactions (
  event_id    uuid PRIMARY KEY,
  user_id     bigint      NOT NULL,
  product_id  bigint      NOT NULL,
  event_type  text        NOT NULL CHECK (event_type IN ('view','cart','purchase')),
  occurred_at timestamptz NOT NULL,
  received_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_interactions_user ON interactions (user_id);
