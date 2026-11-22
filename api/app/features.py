"""Feature store en memoria: arma las 18 características que espera el modelo para
cada par (usuario, producto) candidato. Mismas definiciones que ml/prepare_data.py."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

PAIR_COLS = ["pair_views", "pair_carts", "pair_sessions", "pair_days_active"]
USER_COLS = ["user_events", "user_products", "user_carts", "user_past_purchases"]
PROD_COLS = ["prod_views", "prod_carts", "prod_purchases", "prod_conversion", "prod_cart_rate",
             "price", "price_vs_category", "category_conversion"]


@dataclass
class Candidate:
    product_id: int
    source: str            # "historial" o "popular_en_categoria"


class FeatureStore:
    def __init__(self, path: str):
        self.features: list[str] = json.load(open(os.path.join(path, "feature_names.json")))
        fs = os.path.join(path, "feature_store")
        self.products = pd.read_parquet(os.path.join(fs, "products.parquet")).set_index("product_id")
        self.users = pd.read_parquet(os.path.join(fs, "users.parquet")).set_index("user_id")
        pairs = pd.read_parquet(os.path.join(fs, "pairs.parquet"))
        pairs["last_event_hour"] = pairs.last_event.dt.hour
        pairs["last_event_weekday"] = pairs.last_event.dt.weekday
        self.pairs = pairs.set_index("user_id").sort_index()
        # productos populares por categoría (para usuarios nuevos y para ampliar candidatos)
        p = self.products.assign(score=self.products.prod_purchases * 3 + self.products.prod_carts)
        self.popular = (p.sort_values("score", ascending=False).groupby("top_category").head(20)
                        .reset_index()[["product_id", "top_category", "score"]])
        self.top_sellers = p.sort_values("score", ascending=False).head(50).index.tolist()

    def known_user(self, user_id: int) -> bool:
        return user_id in self.users.index

    def user_history(self, user_id: int) -> pd.DataFrame:
        if user_id not in self.pairs.index:
            return pd.DataFrame(columns=self.pairs.columns)
        h = self.pairs.loc[[user_id]]
        return h

    def candidates(self, user_id: int, max_candidates: int = 50) -> list[Candidate]:
        hist = self.user_history(user_id)
        bought = set(hist[hist.purchased == 1].product_id.astype(int))
        hist = hist[hist.purchased == 0]          # no recomendar lo que ya compró
        out = [Candidate(int(pid), "historial") for pid in hist.product_id.tolist()]
        seen = {c.product_id for c in out} | bought
        cats = self.products.loc[self.products.index.intersection(hist.product_id), "top_category"].value_counts().index.tolist()
        pool = self.popular[self.popular.top_category.isin(cats)].product_id.tolist() if cats else []
        for pid in pool + self.top_sellers:
            if len(out) >= max_candidates:
                break
            if pid not in seen:
                out.append(Candidate(int(pid), "popular_en_categoria" if pid in pool else "mas_vendidos"))
                seen.add(pid)
        return out[:max_candidates]

    def build_rows(self, user_id: int, product_ids: list[int], live: dict[int, dict] | None = None) -> pd.DataFrame:
        """Matriz de características (una fila por producto) en el orden de feature_names.json.
        `live` suma las interacciones recientes que llegaron por RabbitMQ (vistas/carros)."""
        live = live or {}
        hist = self.user_history(user_id).set_index("product_id") if user_id in self.pairs.index else None
        u = self.users.loc[user_id] if user_id in self.users.index else pd.Series(0, index=USER_COLS)
        rows = []
        for pid in product_ids:
            r = {c: 0.0 for c in self.features}
            if hist is not None and pid in hist.index:
                h = hist.loc[pid]
                h = h.iloc[0] if isinstance(h, pd.DataFrame) else h
                for c in PAIR_COLS + ["last_event_hour", "last_event_weekday"]:
                    r[c] = float(h[c])
            lv = live.get(pid, {})
            r["pair_views"] += lv.get("view", 0)
            r["pair_carts"] += lv.get("cart", 0)
            if pid in self.products.index:
                pr = self.products.loc[pid]
                for c in PROD_COLS:
                    r[c] = float(pr[c])
            for c in USER_COLS:
                r[c] = float(u[c])
            rows.append([r[c] for c in self.features])
        return pd.DataFrame(rows, columns=self.features)

    def product_info(self, pid: int) -> dict:
        if pid not in self.products.index:
            return {"category": None, "brand": None, "price": None}
        p = self.products.loc[pid]
        return {"category": p.category_code if isinstance(p.category_code, str) else p.top_category,
                "brand": p.brand, "price": round(float(p.price), 2)}
