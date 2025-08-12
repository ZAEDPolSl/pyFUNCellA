# pyFUNCellA
![Run unit tests](https://github.com/ZAEDPolSl/enrichment-auc/actions/workflows/unittest.yml/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/zaedpolsl/enrichment-auc/badge?s=0a2708157028b922c097a34ac955fe1c363866be)](https://www.codefactor.io/repository/github/zaedpolsl/enrichment-auc)

This repository is a Python implementation of the [FUNCellA R package](https://github.com/ZAEDPolSl/FUNCellA/tree/master).
You can use the original R version or run this Python version with the instructions below.

The presented package provides a wrapper solution for single-sample pathway enrichment algorithms. Additional functionality includes sample-level thresholding.

Implemented single-sample enrichment algorithms:
1) AUCell scoring (Aibar et al. 2017) - scRNA-Seq only
2) BINA (Zyla et al. 2025?) - scRNA-Seq only
3) CERNO AUC (Zyla et al. 2019)
4) JASMINE (Noureen et al. 2022) - scRNA-Seq only
5) Mean
6) ssGSEA (Barbie et al. 2009, Hänzelmann et al. 2013)
7) Z-score (Lee et al. 2008)

Implemented thresholding solutions:
1) AUCell package thresholding (Aibar et al. 2017)
2) k-means
3) GMM thresholding with Top 1 and k-means adjustment (Zyla et al. 2025?)

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

## REFERENCES
Aibar, S., Bravo González-Blas, C., Moerman, T., Huynh-Thu, V.A., Imrichová, H., Hulselmans, G., Rambow, F., Marine, J.C., Geurts, P., Aerts, J., van den Oord, J., Kalender Atak, Z., Wouters, J., & Aerts, S (2017). SCENIC: Single-cell regulatory network inference and clustering. *Nature Methods*, *14*, 1083–1086.\
Barbie, D.A., Tamayo, P., Boehm, J.S., et al. (2009). Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. *Nature*, *462*(7273), 108–112.\
Hänzelmann, S., Castelo, R., & Guinney, J. (2013). GSVA: gene set variation analysis for microarray and RNA-seq data. *BMC Bioinformatics*, *14*, 7.\
Lee, E., Chuang, H.Y., Kim, J.W., Ideker, T., & Lee, D. (2008). Inferring pathway activity toward precise disease classification. *PLoS Computational Biology*, *4*(11), e1000217.\
Noureen, N., Ye, Z., Chen, Y., Wang, X., & Zheng, S. (2022). Signature-scoring methods developed for bulk samples are not adequate for cancer single-cell RNA sequencing data. *Elife*, *11*, e71994.\
Zyla, J., Marczyk, M., Domaszewska, T., Kaufmann, S. H., Polanska, J., & Weiner III, J. (2019). Gene set enrichment for reproducible science: comparison of CERNO and eight other algorithms. *Bioinformatics*, *35*(24), 5146–5154. 
