# Build stage - contains all build dependencies
FROM rocker/r-ver:4.4.2 AS builder

RUN apt-get update &&\
    apt-get install -y \
    software-properties-common \
    curl \
    git \
    gcc \
    g++ \
    build-essential \
    r-base-dev \
    gfortran \
    libgomp1 \
    libgfortran5 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libdeflate-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    libicu-dev \
    libreadline-dev \
    libmagick++-dev \
    pkg-config \
    cmake &&\
    add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils &&\
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3.11 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Install R packages using renv
WORKDIR /app
COPY renv.lock renv/ ./
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org/')" &&\
    R -e "renv::restore()" || echo "Warning: Some R packages failed to install"

# Export Python dependencies and install (production only)
COPY pyproject.toml poetry.lock /app/
RUN poetry export -f requirements.txt --output requirements.txt --without dev &&\
    pip3 install --upgrade pip setuptools wheel &&\
    R_HOME=$(R RHOME) pip3 install -r requirements.txt

# Build the package
COPY README.md /app/
COPY enrichment_auc /app/enrichment_auc
RUN poetry build

# Production stage - minimal image for deployment
FROM rocker/r-ver:4.4.2 AS production

ENV PYTHONUNBUFFERED=TRUE
RUN mkdir -p /root/.config/matplotlib &&\
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

## All necessary runtime dependencies are already present in the copied artifacts from builder

WORKDIR /app

COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY --from=builder /app/dist /app/dist
RUN /app/venv/bin/pip install /app/dist/*.whl
# Install R packages using renv script
COPY renv.lock ./
COPY setup_docker_compatible_renv.R ./
RUN chmod +x setup_docker_compatible_renv.R && Rscript setup_docker_compatible_renv.R

EXPOSE 8050
VOLUME /data
WORKDIR /data
