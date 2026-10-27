# GEFCom2014 month-ahead probabilistic load forecasting (LightGBM).
#
# Default command runs the test suite (synthetic data only, no dataset needed):
#   docker build -t gefcom2014 .
#   docker run --rm gefcom2014
#
# The Kaggle data and the reports are never baked into the image; bind-mount
# them at run time (see docker-compose.yml or the README "Docker and CI").
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# LightGBM's Linux wheel links against the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so source edits do not invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user; pre-create the mount points so they are writable by it.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/src/data/raw /app/src/data/processed /app/reports \
    && chown -R app:app /app

COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
COPY --chown=app:app tests/ tests/

USER app

CMD ["pytest", "-q", "tests/"]
