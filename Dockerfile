FROM python:3.12-slim

ARG TORCH_INDEX_URL=""
ARG TORCH_VERSION="2.13.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY constraints ./constraints
COPY src ./src

RUN python -m pip install -c constraints/py312.txt --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         python -m pip install -c constraints/py312.txt \
           "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"; \
       else \
         python -m pip install -c constraints/py312.txt "torch==$TORCH_VERSION"; \
       fi \
    && python -m pip install -c constraints/py312.txt -e . \
    && python -m pip check

COPY configs ./configs
COPY manifests ./manifests
COPY scripts ./scripts
COPY server ./server

RUN mkdir -p /workspace/data /workspace/runs

ENTRYPOINT ["asgcn-recon"]
CMD ["--help"]
