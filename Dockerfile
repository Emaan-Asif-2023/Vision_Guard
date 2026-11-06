
FROM python:3.12-slim

WORKDIR /app

COPY req-docker.txt .
RUN pip install --no-cache-dir -r req-docker.txt

RUN useradd --create-home appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=5)" || exit 1

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
