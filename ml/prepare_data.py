"""Prepara el dataset de entrenamiento a partir de events.csv (Kaggle: eCommerce events
history in electronics store).

Unidad de predicción: un par (usuario, producto) con al menos una vista o carro.
Etiqueta: 1 si el usuario compró ese producto, 0 si no.

Para evitar fuga de información (data leakage):
  * Las características del par usan solo vistas/carros ANTERIORES a la primera compra.
  * Las estadísticas de producto y de usuario se calculan solo con la ventana de entrenamiento
    (antes de 2021-01-01), y en entrenamiento excluyen el propio par ("leave-one-out").
  * Partición temporal: entrenamiento = pares iniciados antes de 2021-01-01,
    validación = enero 2021, prueba = febrero 2021 (simula producción: el modelo predice el futuro).

Salidas (en --out):
  train.csv, validation.csv, test.csv   -> formato XGBoost de SageMaker: etiqueta primero, sin encabezado
  test_with_ids.parquet                 -> prueba con IDs, para evaluar el endpoint
  feature_store/products.parquet        -> estadísticas por producto (las usa la API)
  feature_store/users.parquet           -> estadísticas por usuario (las usa la API)
  feature_store/pairs.parquet           -> historial usuario-producto (las usa la API)
  feature_names.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURES = [
    "pair_views", "pair_carts", "pair_sessions", "pair_days_active", "last_event_hour", "last_event_weekday",
    "prod_views", "prod_carts", "prod_purchases", "prod_conversion", "prod_cart_rate",
    "price", "price_vs_category", "category_conversion",
    "user_events", "user_products", "user_carts", "user_past_purchases",
]
TRAIN_END = pd.Timestamp("2021-01-01", tz="UTC")
VAL_END = pd.Timestamp("2021-02-01", tz="UTC")


def load(path):
    df = pd.read_csv(path, parse_dates=["event_time"])
    df["top_category"] = df["category_code"].fillna("sin_categoria").str.split(".").str[0]
    df["brand"] = df["brand"].fillna("sin_marca")
    return df


def product_stats(ev):
    g = ev.groupby("product_id")
    st = pd.DataFrame({
        "prod_views": g.event_type.apply(lambda s: (s == "view").sum()),
        "prod_carts": g.event_type.apply(lambda s: (s == "cart").sum()),
        "prod_purchases": g.event_type.apply(lambda s: (s == "purchase").sum()),
    })
    return st


def build_pairs(df):
    """Una fila por par con características previas a la primera compra."""
    first_purchase = (df[df.event_type == "purchase"].groupby(["user_id", "product_id"]).event_time.min()
                      .rename("first_purchase"))
    vc = df[df.event_type != "purchase"].merge(first_purchase, on=["user_id", "product_id"], how="left")
    vc = vc[vc.first_purchase.isna() | (vc.event_time < vc.first_purchase)]
    g = vc.groupby(["user_id", "product_id"])
    pairs = pd.DataFrame({
        "pair_views": g.event_type.apply(lambda s: (s == "view").sum()),
        "pair_carts": g.event_type.apply(lambda s: (s == "cart").sum()),
        "pair_sessions": g.user_session.nunique(),
        "first_event": g.event_time.min(),
        "last_event": g.event_time.max(),
    }).reset_index()
    pairs["pair_days_active"] = (pairs.last_event - pairs.first_event).dt.total_seconds() / 86400
    pairs = pairs.merge(first_purchase.reset_index(), on=["user_id", "product_id"], how="left")
    pairs["label"] = pairs.first_purchase.notna().astype(int)
    # contexto temporal de la última interacción (no depende de la etiqueta)
    pairs["last_event_hour"] = pairs.last_event.dt.hour
    pairs["last_event_weekday"] = pairs.last_event.dt.weekday
    return pairs


def main(events_path, out):
    os.makedirs(os.path.join(out, "feature_store"), exist_ok=True)
    df = load(events_path)
    meta = df.drop_duplicates("product_id", keep="last").set_index("product_id")[["top_category", "category_code", "brand", "price"]]

    hist = df[df.event_time < TRAIN_END]                      # ventana histórica (entrenamiento)
    pst = product_stats(hist).reindex(meta.index).fillna(0)
    cat_median = meta.groupby("top_category").price.median()
    cat_conv = (hist.assign(p=hist.event_type == "purchase", v=hist.event_type == "view")
                .groupby("top_category")[["p", "v"]].sum())
    cat_conv = (cat_conv.p / cat_conv.v.clip(lower=1)).rename("category_conversion")

    ug = hist.groupby("user_id")
    ust = pd.DataFrame({
        "user_events": ug.size(),
        "user_products": ug.product_id.nunique(),
        "user_carts": ug.event_type.apply(lambda s: (s == "cart").sum()),
        "user_past_purchases": ug.event_type.apply(lambda s: (s == "purchase").sum()),
    })

    pairs = build_pairs(df)
    pairs = pairs.merge(meta, left_on="product_id", right_index=True, how="left")
    pairs = pairs.merge(pst, left_on="product_id", right_index=True, how="left")
    pairs = pairs.merge(ust, left_on="user_id", right_index=True, how="left")
    pairs[ust.columns] = pairs[ust.columns].fillna(0)

    in_train = pairs.first_event < TRAIN_END
    # leave-one-out: en entrenamiento se quita el aporte del propio par a las estadísticas
    pairs.loc[in_train, "prod_views"] -= pairs.loc[in_train, "pair_views"]
    pairs.loc[in_train, "prod_carts"] -= pairs.loc[in_train, "pair_carts"]
    lbl_train = in_train & (pairs.first_purchase < TRAIN_END)
    pairs.loc[lbl_train, "prod_purchases"] -= 1
    pairs.loc[lbl_train, "user_past_purchases"] -= 1
    pairs.loc[in_train, "user_events"] -= pairs.loc[in_train, "pair_views"] + pairs.loc[in_train, "pair_carts"] + lbl_train[in_train].astype(int)
    pairs.loc[in_train, "user_carts"] -= pairs.loc[in_train, "pair_carts"]
    pairs.loc[in_train, "user_products"] -= 1
    for c in ["prod_views", "prod_carts", "prod_purchases", "user_past_purchases", "user_events", "user_carts", "user_products"]:
        pairs[c] = pairs[c].clip(lower=0)
    pairs["prod_conversion"] = pairs.prod_purchases / pairs.prod_views.clip(lower=1)
    pairs["prod_cart_rate"] = pairs.prod_carts / pairs.prod_views.clip(lower=1)
    pairs["price_vs_category"] = pairs.price / pairs.top_category.map(cat_median).clip(lower=0.01)
    pairs["category_conversion"] = pairs.top_category.map(cat_conv).fillna(0)

    split = np.where(in_train, "train", np.where(pairs.first_event < VAL_END, "validation", "test"))
    pairs["split"] = split
    for name in ["train", "validation", "test"]:
        part = pairs[pairs.split == name]
        part[["label"] + FEATURES].to_csv(os.path.join(out, f"{name}.csv"), header=False, index=False, float_format="%.6g")
        print(f"{name:10s} filas={len(part):7d} positivos={part.label.sum():6d} ({part.label.mean():.2%})")
    pairs[pairs.split == "test"][["user_id", "product_id", "label"] + FEATURES].to_parquet(os.path.join(out, "test_with_ids.parquet"))

    # ---- feature store para la API (estado "actual": todo el historial disponible)
    full_pst = product_stats(df).reindex(meta.index).fillna(0)
    products = meta.join(full_pst)
    products["prod_conversion"] = products.prod_purchases / products.prod_views.clip(lower=1)
    products["prod_cart_rate"] = products.prod_carts / products.prod_views.clip(lower=1)
    products["price_vs_category"] = products.price / products.top_category.map(cat_median).clip(lower=0.01)
    products["category_conversion"] = products.top_category.map(cat_conv).fillna(0)
    products.reset_index().to_parquet(os.path.join(out, "feature_store", "products.parquet"))
    fg = df.groupby("user_id")
    users = pd.DataFrame({
        "user_events": fg.size(), "user_products": fg.product_id.nunique(),
        "user_carts": fg.event_type.apply(lambda s: (s == "cart").sum()),
        "user_past_purchases": fg.event_type.apply(lambda s: (s == "purchase").sum()),
    })
    users.reset_index().to_parquet(os.path.join(out, "feature_store", "users.parquet"))
    pairs[["user_id", "product_id", "pair_views", "pair_carts", "pair_sessions", "pair_days_active", "last_event", "label"]] \
        .rename(columns={"label": "purchased"}).to_parquet(os.path.join(out, "feature_store", "pairs.parquet"))
    json.dump(FEATURES, open(os.path.join(out, "feature_names.json"), "w"), indent=2)
    print("feature store:", len(products), "productos,", len(users), "usuarios,", len(pairs), "pares")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/raw/events.csv")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    main(a.events, a.out)
