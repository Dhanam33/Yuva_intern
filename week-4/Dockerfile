FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so the layer caches across code changes.
COPY week-4/requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code, static assets and the serialised artifact.
COPY week-4/src ./src
COPY week-4/static ./static
COPY week-4/outputs/model ./outputs/model
COPY week-4/outputs/model_card.json ./outputs/model_card.json
COPY week-4/outputs/model_manifest.json ./outputs/model_manifest.json

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
