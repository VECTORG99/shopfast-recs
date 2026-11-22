"""Prueba de carga de la API ShopFast con Locust.

Local:  locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 2m \
            --csv loadtest/results/local --exclude-tags escritura
AWS:    locust -f loadtest/locustfile.py --host https://api.<ip>.sslip.io --headless -u 50 -r 10 -t 3m \
            --csv loadtest/results/aws --html loadtest/results/aws.html
Variables: API_KEY (por defecto dev-key).
"""
import os
import random

from locust import HttpUser, between, tag, task

KEY = {"X-API-Key": os.getenv("API_KEY", "dev-key")}
USERS = [1515915625532264305, 1515915625537806926, 1515915625572739225, 1515915625595034171,
         1515915625563621633, 1515915625543102921, 999, 12345]
PRODUCTS = [4099648, 3581576, 1405196, 287525, 246841]


class Cliente(HttpUser):
    wait_time = between(0.5, 2)

    @tag("lectura")
    @task(7)
    def recomendaciones(self):
        self.client.get(f"/v1/recommendations/{random.choice(USERS)}?n=10", headers=KEY, name="/v1/recommendations/{user_id}")

    @tag("lectura")
    @task(2)
    def prediccion(self):
        self.client.post("/v1/predict", headers=KEY, json={"user_id": random.choice(USERS), "product_id": random.choice(PRODUCTS)})

    @tag("escritura")
    @task(1)
    def interaccion(self):
        self.client.post("/v1/interactions", headers=KEY, json={"user_id": random.choice(USERS),
                         "product_id": random.choice(PRODUCTS), "event_type": random.choice(["view", "cart"])})
