"""Entrena localmente el mismo XGBoost que se entrena en SageMaker (mismos hiperparámetros)
y calcula las métricas del modelo: accuracy, precision, recall, F1, ROC-AUC y Hit@1 (ranking).

Uso: python ml/train_local.py --data data/processed --out ml/artifacts
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

# Deben coincidir con sagemaker/train_deploy.py
HYPERPARAMS = {
    "objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 6, "eta": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "scale_pos_weight": 5,
}
NUM_ROUND, EARLY_STOP = 400, 25


def read(path, names):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 1:].set_axis(names, axis=1), df.iloc[:, 0].values


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"threshold": round(float(thr), 3), "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def hit_at_1(ids, p):
    """Ranking: entre usuarios con ≥ 2 candidatos y ≥ 1 compra, ¿el producto que el modelo pone
    primero es uno que el usuario compró? Se compara con elegir un candidato al azar."""
    d = ids.assign(p=p)
    g = d.groupby("user_id").label.agg(["size", "sum"])
    ok = g[(g["size"] >= 2) & (g["sum"] > 0)].index
    d = d[d.user_id.isin(ok)].sort_values(["user_id", "p"], ascending=[True, False])
    hit = float(d.groupby("user_id").head(1).label.mean())
    rnd = float((g.loc[ok, "sum"] / g.loc[ok, "size"]).mean())
    return hit, rnd, int(len(ok))


def main(data, out):
    names = json.load(open(os.path.join(data, "feature_names.json")))
    Xtr, ytr = read(os.path.join(data, "train.csv"), names)
    Xva, yva = read(os.path.join(data, "validation.csv"), names)
    Xte, yte = read(os.path.join(data, "test.csv"), names)
    # sin nombres de columnas: igual que el modelo que produce SageMaker (entrada CSV)
    dtr, dva, dte = (xgb.DMatrix(X.values, label=y) for X, y in [(Xtr, ytr), (Xva, yva), (Xte, yte)])
    booster = xgb.train(HYPERPARAMS, dtr, NUM_ROUND, evals=[(dtr, "train"), (dva, "validation")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=50)
    pva = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    pte = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    # umbral que maximiza F1 en validación (nunca en prueba)
    grid = np.arange(0.05, 0.96, 0.01)
    best_thr = max(grid, key=lambda t: f1_score(yva, (pva >= t).astype(int)))
    ids = pd.read_parquet(os.path.join(data, "test_with_ids.parquet"))[["user_id", "product_id", "label"]]
    hit1, hit1_rand, n_users = hit_at_1(ids, pte)
    imp = {names[int(k[1:])]: v for k, v in booster.get_score(importance_type="gain").items()}
    total = sum(imp.values())
    res = {
        "model": "XGBoost binary:logistic", "hyperparameters": HYPERPARAMS, "best_iteration": booster.best_iteration + 1,
        "rows": {"train": len(ytr), "validation": len(yva), "test": len(yte)},
        "test_at_tuned_threshold": metrics_at(yte, pte, best_thr),
        "test_at_0.5": metrics_at(yte, pte, 0.5),
        "roc_auc": roc_auc_score(yte, pte), "pr_auc": average_precision_score(yte, pte),
        "baseline_positive_rate": float(yte.mean()),
        "hit_at_1": hit1, "hit_at_1_random": hit1_rand, "hit_at_1_users": n_users,
        "feature_importance_gain": {k: round(v / total, 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])},
    }
    os.makedirs(out, exist_ok=True)
    booster.save_model(os.path.join(out, "xgboost-model.json"))
    json.dump({"threshold": round(float(best_thr), 2), "features": names}, open(os.path.join(out, "model_config.json"), "w"), indent=2)
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "hyperparameters"}, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="ml/artifacts")
    a = ap.parse_args()
    main(a.data, a.out)
