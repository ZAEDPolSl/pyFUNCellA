# Build stage - contains all build dependencies
FROM rocker/r-ver:4.3.3 AS builder

# Install system dependencies needed for building
RUN apt-get update &&\
    apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgomp1 \
    gcc \
    g++ \
    curl \
    git \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    libicu-dev \
    zlib1g-dev \
    libreadline-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential &&\
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN poetry self add poetry-plugin-export

# Install R packages using renv
WORKDIR /app
COPY renv.lock renv/ ./
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org/')" &&\
    R -e "renv::restore()" || echo "Warning: Some R packages failed to install"

# Export Python dependencies and install
COPY pyproject.toml poetry.lock /app/
RUN poetry export -f requirements.txt --output requirements.txt --with dev &&\
    pip3 install --upgrade pip setuptools wheel &&\
    R_HOME=$(R RHOME) pip3 install -r requirements.txt

# Build the package
COPY README.md /app/
COPY enrichment_auc /app/enrichment_auc
RUN poetry build

# Runtime stage - minimal image for running tests
FROM rocker/r-ver:4.3.3 AS runtime

ENV PYTHONUNBUFFERED=TRUE
RUN mkdir -p /root/.config/matplotlib &&\
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

# Install only runtime dependencies
RUN apt-get update &&\
    apt-get install -y \
    python3.10 \
    python3-pip \
    libgomp1 \
    libpcre2-8-0 \
    liblzma5 \
    libbz2-1.0 \
    libicu70 \
    zlib1g \
    libreadline8 \
    libcurl4 \
    libssl3 \
    libxml2 &&\
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Copy R packages and Python dependencies from builder
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy built package and install
COPY --from=builder /app/dist /app/dist
RUN pip3 install /app/dist/*.whl

# Copy source code and tests
COPY enrichment_auc /app/enrichment_auc
COPY test /app/test

# Run tests
RUN python3 -m pytest
