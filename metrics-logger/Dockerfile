# ds4 metrics logger: polls ds4-server GET /metrics and appends every sample
# to a SQLite database under /data.  Standard library only, no pip install.
#
#   docker run -d --name ds4-metrics --network host \
#     -v ds4-metrics:/data -e DS4_METRICS_URL=http://127.0.0.1:8000 \
#     ghcr.io/<owner>/ds4-metrics-logger:latest
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DS4_METRICS_URL=http://127.0.0.1:8000 \
    DS4_METRICS_DB=/data/ds4-metrics.sqlite \
    DS4_METRICS_INTERVAL=15

RUN useradd --system --uid 10001 --home-dir /data --no-create-home ds4 \
 && mkdir -p /data && chown ds4 /data

COPY metrics_logger.py /app/metrics_logger.py

USER ds4
WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["python3", "/app/metrics_logger.py"]
