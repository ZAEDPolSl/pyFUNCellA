import multiprocessing as mp

import numpy as np
from scipy import stats
from tqdm import tqdm

from enrichment_auc.metrics.gsea import calculate_gene_zscores, rank_expressions


def _calculate_gene_poissons(x):
    x = x.astype(float)
    added = x + 0.5
    for i in tqdm(range(x.shape[0]), mininterval=30):
        x[i, :] = stats.poisson.cdf(x[i, :, np.newaxis], added[i, :]).mean(axis=1)
    return x


def _single_poisson(patient):
    return stats.poisson.cdf(
        patient.reshape((patient.shape[1], 1)), patient + 0.5
    ).mean(axis=1)


def _multi_poisson(arr):
    num_processes = mp.cpu_count()
    with mp.Pool(num_processes) as p:
        chunks = [arr[i : i + 1, :] for i in range(0, arr.shape[0], 1)]
        r = np.array(
            list(
                tqdm(
                    p.imap(
                        _single_poisson,
                        chunks,
                    ),
                    total=arr.shape[0],
                )
            )
        )
    return r


def get_ranks(data, genes, datatype="single_cell", multiprocess=True):
    print("Transforming the geneset...\n")
    if datatype == "single_cell":
        if multiprocess:
            transformed = _multi_poisson(data)
        else:
            transformed = _calculate_gene_poissons(data)
    else:
        bandwidths = (np.std(data, axis=1) / 4)[:, None]
        if np.where(bandwidths == 0)[0].shape[0] != 0:
            good_indices = np.where(bandwidths != 0)[0]
            data = data[good_indices]
            bandwidths = bandwidths[good_indices]
            genes = [genes[i] for i in good_indices.tolist()]
            print(
                "Some of the genes had 0 standard deviation. Removing them from the dataset."
            )
        # divide each row by its bandwidth
        data = data / bandwidths
        transformed = calculate_gene_zscores(data)
    print("Getting the ranks...\n")
    ranks = rank_expressions(transformed)
    return ranks, genes
