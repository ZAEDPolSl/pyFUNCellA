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
    cmake \
    ssh \
    openssl &&\
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
COPY renv.lock ./
COPY setup_docker_compatible_renv.R ./
RUN chmod +x setup_docker_compatible_renv.R && Rscript setup_docker_compatible_renv.R

# Export Python dependencies and install
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-interaction --no-root

# Build the package
COPY README.md /app/
COPY enrichment_auc /app/enrichment_auc
RUN poetry build

# Production stage - minimal image for deployment
FROM rocker/r-ver:4.4.2 AS production

ENV PYTHONUNBUFFERED=TRUE
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    libgomp1 \
    libgfortran5 \
    libjpeg-turbo8 \
    libpng16-16 \
    libfreetype6 \
    libicu-dev \
    libmagick++-6.q16-9t64 \
    libmagick++-6-headers \
    imagemagick \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    libpcre2-dev \
    libdeflate-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
ENV R_LIBS_USER=/usr/local/lib/R/site-library

# Ensure R can find the packages by setting up the library path
RUN mkdir -p /usr/local/lib/R/etc
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"))' >> /usr/local/lib/R/etc/Rprofile.site
RUN echo '.libPaths(c("/usr/local/lib/R/site-library", .libPaths()))' >> /usr/local/lib/R/etc/Rprofile.site

COPY --from=builder /app/dist /app/dist
RUN pip install /app/dist/*.whl

EXPOSE 8050
VOLUME /data
WORKDIR /data
