"""Pruebas unitarias: feature store, predictores, explicaciones y validación del worker."""
import io
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import FEATURES, ROOT, USER_A, USER_NEW

sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, os.path.join(ROOT, "worker"))


@pytest.fixture(scope="module")
def store(data_dir):
    from app.features import FeatureStore
    return FeatureStore(data_dir)


def test_tc01_rows_follow_model_feature_order(store):
    X = store.build_rows(USER_A, [1000, 1001])
    assert list(X.columns) == FEATURES and X.shape == (2, len(FEATURES))


def test_tc02_history_features_are_loaded(store):
    X = store.build_rows(USER_A, [1000])
    assert X.loc[0, "pair_carts"] == 2 and X.loc[0, "pair_views"] == 5 and X.loc[0, "user_past_purchases"] == 2


def test_tc03_live_interactions_add_to_history(store):
    X = store.build_rows(USER_A, [1001], live={1001: {"cart": 2, "view": 1}})
    assert X.loc[0, "pair_carts"] == 2 and X.loc[0, "pair_views"] == 3


def test_tc04_purchased_products_are_not_recommended(store):
    cands = [c.product_id for c in store.candidates(USER_A)]
    assert 1003 not in cands and 1000 in cands


def test_tc05_new_user_gets_popular_candidates_and_zero_user_features(store):
    cands = store.candidates(USER_NEW)
    assert cands and all(c.source == "mas_vendidos" for c in cands)
    X = store.build_rows(USER_NEW, [cands[0].product_id])
    assert X.loc[0, ["user_events", "pair_views", "pair_carts"]].sum() == 0


def test_tc06_local_predictor_returns_probabilities(data_dir, store):
    from app.predictor import LocalPredictor, load_booster
    p = LocalPredictor(load_booster(os.path.join(data_dir, "model", "xgboost-model.json")))
    probs = p.predict(store.build_rows(USER_A, [1000, 1001, 1002]))
    assert probs.shape == (3,) and np.all((probs >= 0) & (probs <= 1))


def test_tc07_explainer_returns_three_readable_reasons(data_dir, store):
    from app.predictor import Explainer, load_booster
    e = Explainer(load_booster(os.path.join(data_dir, "model", "xgboost-model.json")), store.features)
    reasons = e.explain(store.build_rows(USER_A, [1000]))[0]
    assert len(reasons) == 3 and all(r["text"] and r["direction"] in ("aumenta", "disminuye") for r in reasons)


def test_tc08_sagemaker_predictor_sends_csv_and_parses_response(store, monkeypatch):
    from app import predictor as pm
    fake = MagicMock()
    fake.invoke_endpoint.return_value = {"Body": io.BytesIO(b"0.81,0.12")}
    monkeypatch.setitem(sys.modules, "boto3", MagicMock(client=MagicMock(return_value=fake)))
    sp = pm.SageMakerPredictor("shopfast-xgb-endpoint", "us-east-1")
    probs = sp.predict(store.build_rows(USER_A, [1000, 1001]))
    kwargs = fake.invoke_endpoint.call_args.kwargs
    assert kwargs["ContentType"] == "text/csv" and kwargs["Body"].decode().count("\n") == 2
    assert probs.tolist() == pytest.approx([0.81, 0.12])


def test_tc09_worker_rejects_invalid_events():
    import worker
    with pytest.raises(ValueError):
        worker.validate({"event_id": "x", "user_id": 1, "product_id": 2, "event_type": "delete"})
    worker.validate({"event_id": "x", "user_id": 1, "product_id": 2, "event_type": "cart"})


def test_tc10_worker_sends_poison_messages_to_dlq():
    import worker
    ch = MagicMock()
    captured = {}
    ch.basic_consume.side_effect = lambda q, cb: captured.setdefault("cb", cb)
    ch.start_consuming.side_effect = lambda: None
    conn = MagicMock(channel=MagicMock(return_value=ch))
    db = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker.psycopg, "connect", lambda *a, **k: db)
        mp.setattr(worker.pika, "BlockingConnection", lambda *a, **k: conn)
        worker.main()
    cb = captured["cb"]
    cb(ch, MagicMock(delivery_tag=1), None, b"no-es-json")
    ch.basic_reject.assert_called_once_with(1, requeue=False)
    cb(ch, MagicMock(delivery_tag=2), None,
       b'{"event_id":"e1","user_id":1,"product_id":2,"event_type":"cart","occurred_at":"2026-11-21T00:00:00Z"}')
    ch.basic_ack.assert_called_once_with(2)
    assert "ON CONFLICT (event_id) DO NOTHING" in db.execute.call_args.args[0]
