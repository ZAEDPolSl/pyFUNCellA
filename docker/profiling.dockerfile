# Build on the development image
FROM enrichment_auc:dev AS profiling

EXPOSE 8888

# Install additional profiling tools
RUN apt-get update &&\
    apt-get install -y \
    gcc \
    git \
    ssh &&\
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    --no-cache-dir \
    jupyterlab \
    line_profiler \
    memory_profiler

CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=''"]
