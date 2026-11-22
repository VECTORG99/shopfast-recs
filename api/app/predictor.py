"""Predictores intercambiables:
  * SageMakerPredictor: invoca el endpoint de SageMaker (XGBoost integrado, text/csv).
  * LocalPredictor: carga el mismo modelo con la librería xgboost (desarrollo, tests y respaldo).
La explicación (TreeSHAP) siempre se calcula con la copia local del mismo modelo."""
from __future__ import annotations

import io
import os
import tarfile
import time

import numpy as np
import pandas as pd
import xgboost as xgb


def load_booster(path: str) -> xgb.Booster:
    """Acepta xgboost-model.json, el binario 'xgboost-model' o el model.tar.gz de SageMaker."""
    b = xgb.Booster()
    if path.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("xgboost-model"))
            raw = t.extractfile(member).read()
        b.load_model(bytearray(raw))
    else:
        b.load_model(path)
    return b


class LocalPredictor:
    name = "local"

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X.values))


class SageMakerPredictor:
    name = "sagemaker"

    def __init__(self, endpoint_name: str, region: str):
        import boto3
        from botocore.config import Config
        self.endpoint = endpoint_name
        self.client = boto3.client("sagemaker-runtime", region_name=region,
                                   config=Config(retries={"max_attempts": 2}, read_timeout=10, connect_timeout=3))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        buf = io.StringIO()
        X.to_csv(buf, header=False, index=False, float_format="%.6g")
        resp = self.client.invoke_endpoint(EndpointName=self.endpoint, ContentType="text/csv",
                                           Accept="text/csv", Body=buf.getvalue().encode())
        body = resp["Body"].read().decode().strip()
        return np.array([float(v) for v in body.replace("\n", ",").split(",") if v])


class Explainer:
    """Contribuciones TreeSHAP por característica (pred_contribs) traducidas a texto de negocio."""
    TEXT = {
        "pair_carts": "agregó el producto al carro {n} {veces}",
        "pair_views": "vio el producto {n} {veces}",
        "pair_sessions": "lo revisó en {n} {sesiones}",
        "pair_days_active": "lleva {v:.1f} días interesado en el producto",
        "user_past_purchases": "el cliente tiene {n} {compras} anteriores",
        "user_carts": "el cliente ha usado el carro {n} {veces}",
        "user_events": "el cliente registra {v:.0f} interacciones",
        "user_products": "el cliente ha revisado {v:.0f} productos",
        "prod_conversion": "el producto convierte {pct:.1%} de sus vistas en compras",
        "prod_cart_rate": "el {pct:.1%} de quienes lo ven lo agregan al carro",
        "prod_purchases": "el producto suma {v:.0f} ventas",
        "prod_views": "el producto suma {v:.0f} vistas",
        "prod_carts": "el producto ha sido agregado al carro {v:.0f} veces",
        "price": "precio US$ {v:.2f}",
        "price_vs_category": "precio {v:.2f} veces la mediana de su categoría",
        "category_conversion": "la categoría convierte {pct:.1%} de las vistas",
        "last_event_hour": "última visita a las {v:.0f} h",
        "last_event_weekday": "última visita el día {v:.0f} de la semana",
    }

    def __init__(self, booster: xgb.Booster, features: list[str]):
        self.booster, self.features = booster, features

    def explain(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        contribs = self.booster.predict(xgb.DMatrix(X.values), pred_contribs=True)[:, :-1]
        out = []
        for row, c in zip(X.values, contribs):
            idx = np.argsort(-np.abs(c))[:top]
            out.append([{
                "feature": self.features[i],
                "impact": round(float(c[i]), 3),
                "direction": "aumenta" if c[i] > 0 else "disminuye",
                "text": self.TEXT.get(self.features[i], self.features[i]).format(
                    v=row[i], pct=row[i], n=int(round(row[i])),
                    veces="vez" if round(row[i]) == 1 else "veces",
                    sesiones="sesión" if round(row[i]) == 1 else "sesiones distintas",
                    compras="compra" if round(row[i]) == 1 else "compras"),
            } for i in idx])
        return out


def build_predictor(model_path: str):
    booster = load_booster(model_path)
    backend = os.getenv("PREDICTOR", "local")
    if backend == "sagemaker":
        return SageMakerPredictor(os.environ["SAGEMAKER_ENDPOINT"], os.getenv("AWS_REGION", "us-east-1")), booster
    return LocalPredictor(booster), booster


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t0) * 1000
