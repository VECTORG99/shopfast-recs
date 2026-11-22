"""Fixtures: un feature store y un modelo XGBoost pequeños (sintéticos) para que los tests
corran en CI sin el dataset completo ni AWS. RabbitMQ y Postgres se reemplazan por dobles."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = json.load(open(os.path.join(ROOT, "tests", "feature_names.json")))
USER_A, USER_NEW = 111, 999


def make_fixture(d):
    os.makedirs(os.path.join(d, "feature_store"), exist_ok=True)
    os.makedirs(os.path.join(d, "model"), exist_ok=True)
    json.dump(FEATURES, open(os.path.join(d, "feature_names.json"), "w"))
    rng = np.random.default_rng(0)
    pids = list(range(1000, 1030))
    products = pd.DataFrame({
        "product_id": pids, "top_category": ["computers"] * 15 + ["electronics"] * 15,
        "category_code": ["computers.notebook"] * 15 + ["electronics.audio"] * 15, "brand": "acme",
        "price": rng.uniform(10, 500, 30), "prod_views": rng.integers(10, 500, 30), "prod_carts": rng.integers(0, 50, 30),
        "prod_purchases": rng.integers(0, 20, 30)})
    for c, num, den in [("prod_conversion", "prod_purchases", "prod_views"), ("prod_cart_rate", "prod_carts", "prod_views")]:
        products[c] = products[num] / products[den]
    products["price_vs_category"] = 1.0
    products["category_conversion"] = 0.03
    products.to_parquet(os.path.join(d, "feature_store", "products.parquet"))
    pd.DataFrame({"user_id": [USER_A], "user_events": [40], "user_products": [5], "user_carts": [3],
                  "user_past_purchases": [2]}).to_parquet(os.path.join(d, "feature_store", "users.parquet"))
    pd.DataFrame({"user_id": [USER_A] * 4, "product_id": [1000, 1001, 1002, 1003], "pair_views": [5, 2, 1, 3],
                  "pair_carts": [2, 0, 0, 1], "pair_sessions": [2, 1, 1, 1], "pair_days_active": [3.0, 0.1, 0.0, 1.0],
                  "last_event": pd.to_datetime(["2021-02-10"] * 4, utc=True), "purchased": [0, 0, 0, 1]}
                 ).to_parquet(os.path.join(d, "feature_store", "pairs.parquet"))
    # modelo: la probabilidad crece con pair_carts (columna 1), como el modelo real
    X = rng.uniform(0, 5, (600, len(FEATURES)))
    y = (X[:, 1] + rng.normal(0, 0.5, 600) > 2.5).astype(int)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 3}, xgb.DMatrix(X, label=y), 30)
    b.save_model(os.path.join(d, "model", "xgboost-model.json"))
    json.dump({"threshold": 0.5, "features": FEATURES}, open(os.path.join(d, "model", "model_config.json"), "w"))
    json.dump({"roc_auc": 0.97, "test_at_tuned_threshold": {"f1": 0.64}}, open(os.path.join(d, "model", "metrics.json"), "w"))


class FakePublisher:
    def __init__(self):
        self.events, self.fail = [], False

    def publish(self, ev):
        if self.fail:
            raise RuntimeError("rabbit caído")
        self.events.append(ev)


class FakeStore:
    def __init__(self):
        self.live = {}

    def live_counts(self, user_id):
        return self.live.get(user_id, {})

    def stats(self):
        return {"total": 0, "by_type": {}, "recent": []}


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("data"))
    make_fixture(d)
    return d


@pytest.fixture(scope="session")
def app_module(data_dir):
    os.environ.update({"DATA_DIR": data_dir, "API_KEYS": "test-key,rl-key", "RATE_LIMIT": "30/minute", "PREDICTOR": "local"})
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from app import main
    main.publisher = FakePublisher()
    main.interactions = FakeStore()
    return main


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


H = {"X-API-Key": "test-key"}
