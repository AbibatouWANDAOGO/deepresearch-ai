# DeepResearch AI: dashboard and research worker in one image.
# The language model runs in its own container (Ollama, see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# CPU-only PyTorch first (about 200 MB instead of several GB with CUDA), then the rest.
COPY requirements.txt requirements-dev.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-dev.txt

# Unprivileged user; storage and model cache are volumes it owns.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/storage /cache/huggingface \
    && chown -R app:app /app /cache

COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app app ./app
COPY --chown=app:app frontend ./frontend
COPY --chown=app:app eval ./eval
COPY --chown=app:app tests ./tests
COPY --chown=app:app .env.example .gitignore ./

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# Inside the container the dashboard must listen on all interfaces; docker-compose publishes
# the port on 127.0.0.1 only, so it stays reachable from this computer alone.
CMD ["streamlit", "run", "frontend/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
