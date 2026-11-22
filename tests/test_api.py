"""Pruebas de integración de la API REST (FastAPI TestClient) e interoperabilidad (contrato OpenAPI)."""
from conftest import H, USER_A, USER_NEW


def test_tc11_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_tc12_recommendations_require_api_key(client):
    assert client.get(f"/v1/recommendations/{USER_A}").status_code == 401
    assert client.get(f"/v1/recommendations/{USER_A}", headers={"X-API-Key": "mala"}).status_code == 401


def test_tc13_recommendations_sorted_and_explained(client):
    r = client.get(f"/v1/recommendations/{USER_A}?n=5", headers=H)
    assert r.status_code == 200
    items = r.json()["items"]
    probs = [i["purchase_probability"] for i in items]
    assert len(items) == 5 and probs == sorted(probs, reverse=True)
    assert all(len(i["reasons"]) == 3 and i["decision"] in ("alta", "media", "baja") for i in items)
    assert r.headers.get("X-Request-Id")


def test_tc14_invalid_n_returns_422(client):
    assert client.get(f"/v1/recommendations/{USER_A}?n=0", headers=H).status_code == 422
    assert client.get(f"/v1/recommendations/{USER_A}?n=500", headers=H).status_code == 422


def test_tc15_cold_start_user_gets_top_sellers(client):
    body = client.get(f"/v1/recommendations/{USER_NEW}?n=3", headers=H).json()
    assert body["known_user"] is False and {i["source"] for i in body["items"]} == {"mas_vendidos"}


def test_tc16_predict_single_pair(client):
    r = client.post("/v1/predict", headers=H, json={"user_id": USER_A, "product_id": 1000})
    assert r.status_code == 200 and 0 <= r.json()["purchase_probability"] <= 1


def test_tc17_interaction_is_published_to_queue(client, app_module):
    r = client.post("/v1/interactions", headers=H, json={"user_id": USER_A, "product_id": 1001, "event_type": "cart"})
    assert r.status_code == 202
    ev = app_module.publisher.events[-1]
    assert ev["event_id"] == r.json()["event_id"] and ev["event_type"] == "cart"


def test_tc18_invalid_interaction_type_returns_422(client):
    r = client.post("/v1/interactions", headers=H, json={"user_id": 1, "product_id": 2, "event_type": "borrar"})
    assert r.status_code == 422


def test_tc19_queue_down_returns_503(client, app_module):
    app_module.publisher.fail = True
    try:
        r = client.post("/v1/interactions", headers=H, json={"user_id": 1, "product_id": 2, "event_type": "view"})
        assert r.status_code == 503
    finally:
        app_module.publisher.fail = False


def test_tc20_live_cart_raises_probability(client, app_module):
    """Flujo completo: un carro registrado (vía cola → worker → BD) sube la probabilidad del producto."""
    before = client.post("/v1/predict", headers=H, json={"user_id": USER_A, "product_id": 1002}).json()
    app_module.interactions.live[USER_A] = {1002: {"cart": 3}}
    after = client.post("/v1/predict", headers=H, json={"user_id": USER_A, "product_id": 1002}).json()
    app_module.interactions.live.clear()
    assert after["purchase_probability"] > before["purchase_probability"]


def test_tc21_model_failure_returns_503(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.predictor, "predict", lambda X: (_ for _ in ()).throw(RuntimeError("endpoint caído")))
    assert client.get(f"/v1/recommendations/{USER_A}", headers=H).status_code == 503


def test_tc22_rate_limit_returns_429(client):
    codes = [client.get(f"/v1/recommendations/{USER_A}?n=1", headers={"X-API-Key": "rl-key"}).status_code for _ in range(32)]
    assert codes[0] == 200 and 429 in codes


def test_tc23_openapi_contract_documents_all_endpoints(client):
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"])
    assert {"/v1/recommendations/{user_id}", "/v1/predict", "/v1/interactions", "/v1/model", "/health"} <= paths
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
