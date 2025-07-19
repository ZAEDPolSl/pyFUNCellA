# Build stage
FROM rocker/r-ver:4.4.2 AS builder

RUN apt-get update && \
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
    cmake && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3.11 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN poetry self add poetry-plugin-export

WORKDIR /app
COPY renv.lock ./
COPY setup_docker_compatible_renv.R ./
RUN chmod +x setup_docker_compatible_renv.R && Rscript setup_docker_compatible_renv.R

COPY pyproject.toml poetry.lock README.md /app/
COPY enrichment_auc /app/enrichment_auc
RUN python3.11 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip && \
    poetry install --only=main
ENV PATH="/app/venv/bin:$PATH"
RUN poetry build

# Runtime stage
FROM rocker/r-ver:4.4.2 AS app

ENV PYTHONUNBUFFERED=TRUE
RUN mkdir -p /root/.config/matplotlib &&\
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

# Install minimal runtime dependencies
RUN apt-get update &&\
    apt-get install -y \
    software-properties-common \
    libgomp1 \
    libgfortran5 \
    libjpeg-turbo8 \
    libpng16-16 \
    libfreetype6 &&\
    add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y \
    python3.11 \
    python3.11-venv &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy R packages and Python dependencies from builder
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV R_LIBS_USER=/usr/local/lib/R/site-library

# Ensure R can find the packages by setting up the library path
RUN mkdir -p /usr/local/lib/R/etc
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"))' >> /usr/local/lib/R/etc/Rprofile.site
RUN echo '.libPaths(c("/usr/local/lib/R/site-library", .libPaths()))' >> /usr/local/lib/R/etc/Rprofile.site

# Copy built package and install
COPY --from=builder /app/dist /app/dist
RUN /app/venv/bin/pip install /app/dist/*.whl


# Install Streamlit and any extra dependencies for the app
RUN /app/venv/bin/pip install streamlit

COPY streamlit_app.py /app/streamlit_app.py

EXPOSE 8501

CMD ["/app/venv/bin/streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]