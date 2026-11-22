.PHONY: data train-local test up up-aws down loadtest-local

data:            ## Genera train/validation/test y el feature store desde data/raw/events.csv
	python ml/prepare_data.py --events data/raw/events.csv --out data/processed

train-local:     ## Entrena el mismo XGBoost localmente y calcula métricas
	python ml/train_local.py --data data/processed --out ml/artifacts
	mkdir -p data/processed/model && cp ml/artifacts/*.json data/processed/model/

test:            ## Pruebas unitarias y de integración
	pytest tests -q

up:              ## Levanta todo localmente (modelo local)
	docker compose up -d --build

up-aws:          ## Levanta en el EC2 con logs a CloudWatch
	docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d --build

down:
	docker compose down

loadtest-local:  ## Prueba de carga local (sin escrituras)
	locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 90s \
	  --csv loadtest/results/local --exclude-tags escritura
