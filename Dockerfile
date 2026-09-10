# syntax=docker/dockerfile:1
#
# One Dockerfile for CPU and CUDA. Swap the base and the torch index:
#
#   CPU (default):
#     docker build -t flysim:cpu .
#
#   CUDA:
#     docker build -t flysim:gpu \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 .
#
# Behind a restrictive corporate proxy (no apt, no public PyPI) use
# Dockerfile.offline instead - see README "Building behind a proxy".
#
# NOT VERIFIED: the CUDA variant was never executed during development (the dev
# host is Apple Silicon with no NVIDIA GPU). See README "Verified vs not".

ARG BASE_IMAGE=python:3.12-trixie

# ---------------------------------------------------------------- deps -----
FROM ${BASE_IMAGE} AS deps
ARG TORCH_INDEX=""
ARG PIP_INDEX_URL=""
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# libegl1 + MUJOCO_GL=egl is how FlyGym's own image does headless rendering;
# ffmpeg is only needed if you want video files out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 ffmpeg ca-certificates \
    && if ! command -v python3 >/dev/null; then \
        apt-get install -y --no-install-recommends python3 python3-venv python3-pip; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
# torch is not in requirements.txt: the ordinary PyPI wheel bundles CUDA even
# for CPU-only use, on aarch64 as well as x86_64 - measured, it is the whole
# difference between a 2 GB image and an 8 GB one. The +cpu index has wheels
# for both, so it is the default here.
#
# The default lives in this RUN and not only in the ARG because docker-compose
# passes `TORCH_INDEX: ${TORCH_INDEX:-}`, and an explicitly empty build-arg
# OVERRIDES an ARG default. With the default only on the ARG this became
# `pip install --index-url ""` and `docker compose build` failed outright.
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip \
    && if [ -n "$PIP_INDEX_URL" ]; then pip config set global.index-url "$PIP_INDEX_URL"; fi \
    && TORCH_INDEX="${TORCH_INDEX:-$TORCH_CPU_INDEX}" \
    && echo "torch index: $TORCH_INDEX ($(uname -m))" \
    && pip install --index-url "$TORCH_INDEX" \
        ${PIP_INDEX_URL:+--extra-index-url "$PIP_INDEX_URL"} torch==2.14.0 \
    && pip install -r /tmp/requirements.txt

# ---------------------------------------------------------- data-fetch -----
# A separate stage on purpose. It copies ONLY brain/, so editing run.py or the
# web UI does not invalidate this layer and force a re-download of connectome
# tables. Fails soft: an image built with no neuPrint access still runs, either
# fetching on first use or with --connectome synthetic.
FROM deps AS data-fetch
ARG CONNECTOME_SUBSET=motor
WORKDIR /app
COPY brain/ /app/brain/
RUN mkdir -p /data/connectome \
    && python3 -m brain.fetch --subset "${CONNECTOME_SUBSET}" --out /data/connectome \
       || echo "connectome prefetch skipped; run.py will fetch on demand"

# ------------------------------------------------------------- runtime -----
FROM deps AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    FLY_ENV_DIR=/app/environments

RUN useradd --create-home --uid 1000 fly \
    && mkdir -p /data/connectome /data/checkpoints /data/runs \
    && chown -R fly:fly /data

WORKDIR /app
COPY --chown=fly:fly . /app
COPY --from=data-fetch --chown=fly:fly /data/connectome /data/connectome

USER fly
CMD ["python3", "run.py", "--list-envs"]
