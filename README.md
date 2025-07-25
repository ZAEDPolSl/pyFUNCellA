# pyFUNCellA
![Run unit tests](https://github.com/ZAEDPolSl/enrichment-auc/actions/workflows/unittest.yml/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/zaedpolsl/enrichment-auc/badge?s=0a2708157028b922c097a34ac955fe1c363866be)](https://www.codefactor.io/repository/github/zaedpolsl/enrichment-auc)

This repository is a Python implementation of the [FUNCellA R package](https://github.com/ZAEDPolSl/FUNCellA/tree/master).
You can use the original R version or run this Python version with the instructions below.

## Installation
 For installation, you have several options - you can build it yourself locally, you can use Docker for automatic build. Alternatively, you can use the application - either on your computer or on remote server.
### Docker (recommended)

Build and run the deployment container:

```bash
docker build -f docker/deploy.dockerfile -t funcella-deploy .
```

### Manual installation
1. renv (R dependencies)

Open R and run:

```R
install.packages("renv")
renv::restore()
```

2. Poetry (Python dependencies)

Install [Poetry](https://python-poetry.org/docs/#installation):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install Python dependencies:

```bash
poetry install
```

### Running the App

```bash
docker build -f docker/streamlit.dockerfile -t funcella-app .
docker run -p 8501:8501 funcella-app
```
The application will be available at your localhost (localhost:8501).
Alternatively, the remote version is available [here]().