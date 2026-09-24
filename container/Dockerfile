# hvr-cnn CPU image. Build from the repository root:
#   podman build --platform linux/amd64 -f container/Dockerfile \
#       --build-arg VERSION=... --build-arg REVISION=$(git rev-parse HEAD) -t hvr-cnn .
#
# DRAFT: base image tag/digest proposed, not yet approved or built.
ARG BASE=docker.io/mambaorg/micromamba:2.9.0-debian13-slim@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d
FROM ${BASE}

# The same explicit lock file the development env is created from.
# Install and clean in one layer so the package cache never reaches the image.
COPY --chown=$MAMBA_USER:$MAMBA_USER container/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN micromamba install -y -n base -f /tmp/conda-linux-64.lock \
    && micromamba clean --all --yes \
    && rm /tmp/conda-linux-64.lock

COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml LICENSE NOTICE /tmp/hvr-cnn/
COPY --chown=$MAMBA_USER:$MAMBA_USER src /tmp/hvr-cnn/src
RUN /opt/conda/bin/python -m pip install --no-deps --no-build-isolation --no-cache-dir /tmp/hvr-cnn \
    && rm -rf /tmp/hvr-cnn \
    && /opt/conda/bin/hvr-cnn --version

# No activated shell at run time: hvr-cnn derives the MINC variables from its
# own prefix (hvr_cnn.minctools.tool_env). These are for interactive use only.
# The image is read-only and runs as an arbitrary uid, so never write bytecode.
ENV PATH=/opt/conda/pipeline:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    MINC_TOOLKIT=/opt/conda \
    PERL5LIB=/opt/conda/perl:/opt/conda/pipeline \
    MNI_DATAPATH=/opt/conda/share \
    ANTSPATH=/opt/conda/bin \
    MINC_FORCE_V2=1 \
    MINC_COMPRESS=4 \
    VOLUME_CACHE_THRESHOLD=-1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG VERSION=0.0.0.dev0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="hvr-cnn" \
      org.opencontainers.image.description="Hippocampus / temporal-horn segmentation and hippocampal-to-ventricle ratio" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="GPL-3.0-only AND CC-BY-4.0" \
      org.opencontainers.image.source="https://github.com/soffiafdz/hvr-cnn"

# Absolute path: must not depend on the working directory, PATH or the user.
ENTRYPOINT ["/opt/conda/bin/hvr-cnn"]
CMD ["--help"]
