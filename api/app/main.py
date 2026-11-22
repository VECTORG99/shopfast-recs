"""ShopFast Recommendation API (FastAPI).

Backend intermedio entre el dashboard/e-commerce y el modelo en SageMaker: valida la petición,
arma las características, invoca el endpoint, ordena el Top-N y explica cada recomendación.
Documentación OpenAPI/Swagger automática en /docs.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from .features import FeatureStore
from .messaging import InteractionStore, Publisher
from .predictor import Explainer, build_predictor, timed

DATA_DIR = os.getenv("DATA_DIR", "/data")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DATA_DIR, "model", "xgboost-model.json"))
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "dev-key").split(",") if k.strip()}
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

store = FeatureStore(DATA_DIR)
predictor, booster = build_predictor(MODEL_PATH)
explainer = Explainer(booster, store.features)
cfg = json.load(open(os.path.join(DATA_DIR, "model", "model_config.json")))
THRESHOLD = float(cfg["threshold"])
METRICS = json.load(open(os.path.join(DATA_DIR, "model", "metrics.json")))
publisher = Publisher()
interactions = InteractionStore()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)) -> str:
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="API Key inválida o ausente (cabecera X-API-Key)")
    return key


def rate_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")


limiter = Limiter(key_func=rate_key, default_limits=[RATE_LIMIT])
app = FastAPI(title="ShopFast Recommendation API", version="1.0.0",
              description="Recomendaciones de productos con XGBoost desplegado en Amazon SageMaker. "
                          "Autenticación con cabecera `X-API-Key`; límite de solicitudes por llave.")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": f"Límite de solicitudes excedido ({exc.detail})"})


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    resp.headers["X-Request-Id"] = rid
    log.info(json.dumps({"request_id": rid, "method": request.method, "path": request.url.path,
                         "status": resp.status_code, "latency_ms": round(ms, 1), "predictor": predictor.name}))
    return resp


# ---------------------------------------------------------------- esquemas
class Reason(BaseModel):
    feature: str
    impact: float
    direction: str
    text: str


class Recommendation(BaseModel):
    rank: int
    product_id: int
    category: str | None
    brand: str | None
    price: float | None
    purchase_probability: float = Field(..., description="Probabilidad de compra estimada por el modelo")
    decision: Literal["alta", "media", "baja"]
    source: str
    reasons: list[Reason]


class RecommendationResponse(BaseModel):
    user_id: str
    known_user: bool
    model_backend: str
    inference_ms: float
    items: list[Recommendation]


class PredictRequest(BaseModel):
    user_id: int = Field(..., examples=[1515915625532264305])
    product_id: int = Field(..., examples=[1405196])


class PredictResponse(BaseModel):
    user_id: str
    product_id: int
    purchase_probability: float
    decision: Literal["alta", "media", "baja"]
    threshold: float
    reasons: list[Reason]


class InteractionIn(BaseModel):
    user_id: int
    product_id: int
    event_type: Literal["view", "cart", "purchase"]


def decision(p: float) -> str:
    return "alta" if p >= THRESHOLD else ("media" if p >= THRESHOLD / 2 else "baja")


# ---------------------------------------------------------------- endpoints
@app.get("/health", tags=["operación"])
@limiter.exempt
def health():
    return {"status": "ok", "predictor": predictor.name,
            "endpoint": os.getenv("SAGEMAKER_ENDPOINT") if predictor.name == "sagemaker" else None,
            "features": len(store.features), "threshold": THRESHOLD}


@app.get("/v1/recommendations/{user_id}", response_model=RecommendationResponse, tags=["recomendaciones"])
def recommendations(user_id: int, request: Request, n: int = 10, _: str = Depends(require_key)):
    if not 1 <= n <= 50:
        raise HTTPException(status_code=422, detail="n debe estar entre 1 y 50")
    cands = store.candidates(user_id)
    live = interactions.live_counts(user_id)
    X = store.build_rows(user_id, [c.product_id for c in cands], live)
    try:
        probs, ms = timed(predictor.predict, X)
    except Exception as exc:                                   # fallback: más vendidos sin modelo
        log.info(json.dumps({"event": "predictor_error", "error": str(exc)}))
        raise HTTPException(status_code=503, detail="Modelo no disponible, intente nuevamente")
    order = sorted(range(len(cands)), key=lambda i: -probs[i])[:n]
    reasons = explainer.explain(X.iloc[order])
    items = []
    for rank, (i, rs) in enumerate(zip(order, reasons), start=1):
        info = store.product_info(cands[i].product_id)
        items.append(Recommendation(rank=rank, product_id=cands[i].product_id, purchase_probability=round(float(probs[i]), 4),
                                    decision=decision(float(probs[i])), source=cands[i].source, reasons=rs, **info))
    return RecommendationResponse(user_id=str(user_id), known_user=store.known_user(user_id),
                                  model_backend=predictor.name, inference_ms=round(ms, 1), items=items)


@app.post("/v1/predict", response_model=PredictResponse, tags=["recomendaciones"])
def predict(body: PredictRequest, request: Request, _: str = Depends(require_key)):
    X = store.build_rows(body.user_id, [body.product_id], interactions.live_counts(body.user_id))
    p = float(predictor.predict(X)[0])
    return PredictResponse(user_id=str(body.user_id), product_id=body.product_id, purchase_probability=round(p, 4),
                           decision=decision(p), threshold=THRESHOLD, reasons=explainer.explain(X)[0])


@app.post("/v1/interactions", status_code=202, tags=["interacciones"])
def track(body: InteractionIn, request: Request, _: str = Depends(require_key)):
    event = {"event_id": str(uuid.uuid4()), "user_id": body.user_id, "product_id": body.product_id,
             "event_type": body.event_type, "occurred_at": datetime.now(timezone.utc).isoformat()}
    try:
        publisher.publish(event)
    except Exception:
        raise HTTPException(status_code=503, detail="Cola de mensajes no disponible")
    return {"status": "accepted", "event_id": event["event_id"]}


@app.get("/v1/model", tags=["modelo"])
def model_info(request: Request, _: str = Depends(require_key)):
    return METRICS


@app.get("/v1/stats", tags=["operación"])
def stats(request: Request, _: str = Depends(require_key)):
    try:
        return interactions.stats()
    except Exception:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
