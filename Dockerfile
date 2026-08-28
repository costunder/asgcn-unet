FROM python:3.12-slim

ARG TORCH_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         python -m pip install torch --index-url "$TORCH_INDEX_URL"; \
       else \
         python -m pip install torch; \
       fi \
    && python -m pip install -e ".[eval]"

COPY configs ./configs
COPY manifests ./manifests
COPY scripts ./scripts
COPY server ./server
COPY tests ./tests

RUN mkdir -p /workspace/data /workspace/runs

ENTRYPOINT ["asgcn-recon"]
CMD ["--help"]
