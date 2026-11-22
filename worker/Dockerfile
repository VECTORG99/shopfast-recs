FROM python:3.11-slim
RUN useradd --uid 10001 --no-create-home app
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
USER 10001
CMD ["python", "worker.py"]
