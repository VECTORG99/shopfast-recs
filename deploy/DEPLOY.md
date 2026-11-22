# Despliegue en AWS Academy Learner Lab (paso a paso)

Arquitectura desplegada: **SageMaker AI** (training job + endpoint del XGBoost integrado) + **un EC2**
con Docker Compose (Caddy HTTPS, API FastAPI, dashboard Streamlit, RabbitMQ, worker y PostgreSQL).
Región: **us-east-1**. Rol: **LabRole** (Learner Lab no permite crear roles ni usuarios IAM).

Tiempo estimado: 60–90 min la primera vez. Costo de una sesión de demo (4 h): < US$ 1.

---

## 0. Antes de empezar
- Inicia el lab (**Start Lab**) y espera el punto verde. Abre la consola en **us-east-1**.
- Ten a mano el archivo `events.csv` del dataset (Caso 1).

## 1. Crear el EC2
Consola → EC2 → **Launch instance**:

| Campo | Valor |
|---|---|
| Nombre | `shopfast-ep3` |
| AMI | Amazon Linux 2023 |
| Tipo | `t3.medium` (2 vCPU, 4 GB) |
| Key pair | `vockey` (descárgala desde *AWS Details* del lab) |
| Red | Security group nuevo: SSH 22 (My IP), HTTP 80 y HTTPS 443 (0.0.0.0/0) |
| Almacenamiento | 20 GB gp3 |
| Advanced → IAM instance profile | **LabInstanceProfile** |
| Advanced → Metadata response hop limit | **2** (para que los contenedores obtengan las credenciales del rol) |
| Advanced → User data | contenido de `deploy/ec2_user_data.sh` |

Luego **Elastic IPs → Allocate → Associate** a la instancia. Así la URL no cambia cuando el lab se reinicia.

## 2. Preparar el código y los datos (en el EC2)
```bash
ssh -i labsuser.pem ec2-user@<IP_ELASTICA>
git clone https://github.com/<usuario>/shopfast-recs.git && cd shopfast-recs
# subir el dataset desde tu PC (en otra terminal):
#   scp -i labsuser.pem events.csv ec2-user@<IP_ELASTICA>:~/shopfast-recs/data/raw/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt
make data            # ~4 min: genera train/validation/test y el feature store
make train-local     # ~10 s: métricas de referencia (ml/artifacts/metrics.json)
```

## 3. Entrenar y desplegar en SageMaker
```bash
export BUCKET=shopfast-ep3-$(date +%s)
python sagemaker/train_deploy.py upload --bucket $BUCKET
python sagemaker/train_deploy.py train  --bucket $BUCKET        # ~5 min (ml.m5.large)
python sagemaker/train_deploy.py deploy --mode serverless       # ~5–8 min
# Si el lab rechaza serverless:
#   python sagemaker/train_deploy.py deploy --mode realtime && python sagemaker/train_deploy.py autoscale
python sagemaker/train_deploy.py evaluate --sample 5000         # métricas medidas contra el endpoint
python sagemaker/train_deploy.py download-model                 # copia local para las explicaciones (SHAP)
```

## 4. Levantar los servicios
```bash
cp .env.example .env
# Editar .env:
#   PREDICTOR=sagemaker
#   MODEL_PATH=/data/model/model.tar.gz
#   DOMAIN=<IP-con-guiones>.sslip.io      (ej. 3-90-12-34.sslip.io)
#   API_KEYS, DASHBOARD_API_KEY y contraseñas: valores propios
make up-aws
docker compose ps        # todos "healthy" / "running"
```

URLs (HTTPS automático con Let's Encrypt; el primer acceso tarda ~30 s mientras se emite el certificado):

| Servicio | URL |
|---|---|
| Dashboard | `https://<DOMAIN>` |
| API + Swagger | `https://api.<DOMAIN>/docs` |
| RabbitMQ (consola) | `https://rabbit.<DOMAIN>` |
| Endpoint del modelo | SageMaker → Inference → Endpoints → `shopfast-xgb-endpoint` |

## 5. Pruebas en la nube
```bash
curl https://api.<DOMAIN>/health
curl -H "X-API-Key: <llave>" "https://api.<DOMAIN>/v1/recommendations/1515915625532264305?n=5"
API_KEY=<llave> locust -f loadtest/locustfile.py --host https://api.<DOMAIN> --headless -u 50 -r 10 -t 3m \
   --csv loadtest/results/aws --html loadtest/results/aws.html
```
> Para la prueba de carga sube `RATE_LIMIT` en `.env` (por ejemplo `10000/minute`) y reinicia la API; vuelve a `60/minute` al terminar.

## 6. Evidencias para el informe (capturas)
1. SageMaker → Training jobs → el job **Completed** con sus métricas (`validation:aucpr`).
2. SageMaker → Endpoints → `shopfast-xgb-endpoint` **InService** (y su configuración serverless o auto scaling).
3. CloudWatch → Metrics → AWS/SageMaker → `Invocations`, `ModelLatency`, `OverheadLatency` del endpoint.
4. CloudWatch → Log groups → `/shopfast/ep3` (logs JSON de la API y del worker) y `/aws/sagemaker/Endpoints/...`.
5. EC2 → instancia **Running** + pestaña Monitoring (CPU).
6. Swagger en `https://api.<DOMAIN>/docs` y una llamada exitosa desde Postman/curl.
7. Dashboard: recomendaciones con explicación + pestaña "Calidad del modelo".
8. RabbitMQ: cola `interactions.persist` con mensajes procesados.
9. Reporte HTML de Locust (`loadtest/results/aws.html`).
10. Navegador mostrando el candado HTTPS.

## 7. Apagar (evita gastar el crédito)
```bash
python sagemaker/train_deploy.py cleanup     # borra endpoint, configuración y modelo
docker compose down                          # opcional
```
Detén el EC2 desde la consola. Antes de la defensa: `deploy` de nuevo (5–8 min) y `make up-aws`.

## Problemas comunes
| Síntoma | Solución |
|---|---|
| La API no obtiene credenciales (`NoCredentialsError`) | Metadata hop limit = 2 y perfil LabInstanceProfile asignado |
| `AccessDenied` al crear el endpoint serverless | Usa `--mode realtime` (instancia `ml.t2.medium`) |
| Certificado HTTPS no se emite | Revisa que 80/443 estén abiertos y que `DOMAIN` coincida con la IP pública |
| La URL cambió tras reiniciar el lab | Asocia una Elastic IP (paso 1) |
| Primera predicción lenta | Cold start del endpoint serverless (~5–10 s); se calienta con una llamada previa |
