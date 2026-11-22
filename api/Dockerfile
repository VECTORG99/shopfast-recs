FROM python:3.11-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home app
COPY --from=build /install /usr/local
WORKDIR /srv
COPY app/ app/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
