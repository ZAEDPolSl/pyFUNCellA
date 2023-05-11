FROM python:3.9-slim AS base
ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y \
    libgomp1 \
    gcc \
    curl \
    git \
    ssh &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

COPY pyproject.toml poetry.lock /app/
COPY setup.py build.py README.md /app/
COPY gamred_native /app/gamred_native
COPY enrichment_auc /app/enrichment_auc
RUN poetry config virtualenvs.create false &&\
    poetry install --with dev &&\
    poetry build
