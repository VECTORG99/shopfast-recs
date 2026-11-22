"""Entrenamiento y despliegue del modelo en Amazon SageMaker AI con boto3 (sin SDK de SageMaker).

Pensado para AWS Academy Learner Lab: usa el rol LabRole, región us-east-1 y el algoritmo
integrado XGBoost 1.7-1 (sin contenedor propio). Ejecutar desde el EC2 (perfil LabInstanceProfile)
o desde CloudShell.

  python sagemaker/train_deploy.py upload    --bucket shopfast-ep3-<sufijo>
  python sagemaker/train_deploy.py train     --bucket shopfast-ep3-<sufijo>
  python sagemaker/train_deploy.py deploy    --mode serverless      # o --mode realtime
  python sagemaker/train_deploy.py autoscale                        # solo modo realtime
  python sagemaker/train_deploy.py evaluate  --sample 5000
  python sagemaker/train_deploy.py download-model
  python sagemaker/train_deploy.py cleanup                          # borra endpoint y config (evita costos)
"""
import argparse
import io
import json
import os
import time

import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")


def xgb_image():
    """URI del algoritmo integrado XGBoost 1.7-1. Se resuelve con el SDK si está instalado
    (v2: sagemaker.image_uris; v3: sagemaker.core.image_uris); si no, usa la ruta conocida de us-east-1."""
    if os.getenv("XGB_IMAGE"):
        return os.environ["XGB_IMAGE"]
    for mod in ("sagemaker.core", "sagemaker"):
        try:
            return __import__(mod, fromlist=["image_uris"]).image_uris.retrieve("xgboost", REGION, "1.7-1")
        except Exception:
            pass
    return "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


XGB_IMAGE = xgb_image()
ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "shopfast-xgb-endpoint")
STATE = os.path.join(os.path.dirname(__file__), ".state.json")
DATA = os.getenv("DATA_DIR", "data/processed")

# Mismos hiperparámetros que ml/train_local.py
HYPERPARAMS = {"objective": "binary:logistic", "eval_metric": "aucpr", "num_round": "400", "early_stopping_rounds": "25",
               "max_depth": "6", "eta": "0.1", "subsample": "0.8", "colsample_bytree": "0.8",
               "min_child_weight": "5", "scale_pos_weight": "5"}

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def state(**kw):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    st.update(kw)
    json.dump(st, open(STATE, "w"), indent=2)
    return st


def role_arn():
    acct = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return os.getenv("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{acct}:role/LabRole")


def upload(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    for name in ["train", "validation", "test"]:
        s3.upload_file(os.path.join(DATA, f"{name}.csv"), bucket, f"data/{name}/{name}.csv")
        print("subido", f"s3://{bucket}/data/{name}/{name}.csv")
    state(bucket=bucket)


def train(bucket):
    job = f"shopfast-xgb-{time.strftime('%Y%m%d-%H%M%S')}"
    ch = lambda n: {"ChannelName": n, "ContentType": "text/csv", "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/data/{n}/", "S3DataDistributionType": "FullyReplicated"}}}
    sm.create_training_job(
        TrainingJobName=job, RoleArn=role_arn(), HyperParameters=HYPERPARAMS,
        AlgorithmSpecification={"TrainingImage": XGB_IMAGE, "TrainingInputMode": "File"},
        InputDataConfig=[ch("train"), ch("validation")],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/models/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        Tags=[{"Key": "project", "Value": "shopfast-ep3"}])
    print("training job:", job)
    sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=job)
    d = sm.describe_training_job(TrainingJobName=job)
    print("estado:", d["TrainingJobStatus"], "| segundos facturables:", d.get("BillableTimeInSeconds"))
    for m in d.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']}: {m['Value']:.4f}")
    state(job=job, model_data=d["ModelArtifacts"]["S3ModelArtifacts"])


def deploy(mode):
    st = state()
    name = f"shopfast-xgb-{time.strftime('%Y%m%d-%H%M%S')}"
    sm.create_model(ModelName=name, ExecutionRoleArn=role_arn(),
                    PrimaryContainer={"Image": XGB_IMAGE, "ModelDataUrl": st["model_data"]})
    variant = {"VariantName": "AllTraffic", "ModelName": name}
    if mode == "serverless":
        variant["ServerlessConfig"] = {"MemorySizeInMB": 2048, "MaxConcurrency": 5}
    else:
        variant.update({"InstanceType": os.getenv("ENDPOINT_INSTANCE", "ml.t2.medium"), "InitialInstanceCount": 1})
    sm.create_endpoint_config(EndpointConfigName=name, ProductionVariants=[variant])
    existing = [e["EndpointName"] for e in sm.list_endpoints(NameContains=ENDPOINT)["Endpoints"]]
    if ENDPOINT in existing:
        sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=name)
    else:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=name)
    print(f"desplegando {ENDPOINT} ({mode})… (5–10 min)")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT)
    print("endpoint InService:", ENDPOINT)
    state(model_name=name, endpoint_config=name, mode=mode)


def autoscale(min_cap=1, max_cap=2, target=100):
    """Auto Scaling del endpoint real-time: escala por invocaciones por instancia."""
    aas = boto3.client("application-autoscaling", region_name=REGION)
    rid = f"endpoint/{ENDPOINT}/variant/AllTraffic"
    aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=rid,
                                 ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                                 MinCapacity=min_cap, MaxCapacity=max_cap)
    aas.put_scaling_policy(PolicyName="shopfast-invocations", ServiceNamespace="sagemaker", ResourceId=rid,
                           ScalableDimension="sagemaker:variant:DesiredInstanceCount", PolicyType="TargetTrackingScaling",
                           TargetTrackingScalingPolicyConfiguration={
                               "TargetValue": float(target), "ScaleOutCooldown": 60, "ScaleInCooldown": 300,
                               "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"}})
    print(f"auto scaling {min_cap}-{max_cap} instancias, objetivo {target} invocaciones/instancia/min")


def evaluate(sample):
    """Envía filas del conjunto de prueba al endpoint y calcula accuracy/precision/recall/F1."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    df = pd.read_csv(os.path.join(DATA, "test.csv"), header=None).sample(sample, random_state=7)
    thr = json.load(open(os.path.join(DATA, "model", "model_config.json")))["threshold"]
    probs, lat = [], []
    for i in range(0, len(df), 500):
        buf = io.StringIO()
        df.iloc[i:i + 500, 1:].to_csv(buf, header=False, index=False)
        t0 = time.perf_counter()
        r = rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Body=buf.getvalue())
        lat.append((time.perf_counter() - t0) * 1000)
        probs += [float(v) for v in r["Body"].read().decode().replace("\n", ",").split(",") if v]
    y, yhat = df[0].values, [int(p >= thr) for p in probs]
    res = {"endpoint": ENDPOINT, "rows": len(y), "threshold": thr, "accuracy": accuracy_score(y, yhat),
           "precision": precision_score(y, yhat), "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
           "roc_auc": roc_auc_score(y, probs), "batch_latency_ms_avg": sum(lat) / len(lat)}
    json.dump(res, open(os.path.join(DATA, "model", "endpoint_metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


def download_model():
    st = state()
    bucket, key = st["model_data"].replace("s3://", "").split("/", 1)
    out = os.path.join(DATA, "model", "model.tar.gz")
    s3.download_file(bucket, key, out)
    print("modelo descargado en", out, "→ use MODEL_PATH=/data/model/model.tar.gz")


def cleanup():
    st = state()
    for fn, kw in [(sm.delete_endpoint, {"EndpointName": ENDPOINT}),
                   (sm.delete_endpoint_config, {"EndpointConfigName": st.get("endpoint_config", "")}),
                   (sm.delete_model, {"ModelName": st.get("model_name", "")})]:
        try:
            fn(**kw)
            print("eliminado:", list(kw.values())[0])
        except Exception as exc:
            print("omitido:", exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "train", "deploy", "autoscale", "evaluate", "download-model", "cleanup"])
    ap.add_argument("--bucket")
    ap.add_argument("--mode", choices=["serverless", "realtime"], default="serverless")
    ap.add_argument("--sample", type=int, default=5000)
    a = ap.parse_args()
    {"upload": lambda: upload(a.bucket), "train": lambda: train(a.bucket or state()["bucket"]),
     "deploy": lambda: deploy(a.mode), "autoscale": autoscale, "evaluate": lambda: evaluate(a.sample),
     "download-model": download_model, "cleanup": cleanup}[a.cmd]()
