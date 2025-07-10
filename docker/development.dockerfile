FROM python:3.10-slim AS base
ENV PYTHONUNBUFFERED=TRUE
RUN mkdir -p /root/.config/matplotlib &&\
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y \
    libgomp1 \
    gcc \
    curl \
    git \
    ssh \
    openssl \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    r-base &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

COPY pyproject.toml poetry.lock /app/
COPY README.md /app/
COPY enrichment_auc /app/enrichment_auc
RUN poetry config virtualenvs.create false &&\
    poetry install --with dev &&\
    poetry build

RUN R -e "install.packages('renv', repos = c(CRAN = 'https://cloud.r-project.org'))"
COPY renv.lock renv.lock
COPY .Rprofile .Rprofile
RUN mkdir -p renv
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json
RUN R -e "renv::restore()"
