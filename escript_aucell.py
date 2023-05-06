import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from enrichment_auc.distributions import (
    find_distribution, find_grouped_dist_thresholds, group_distributions)


def pipeline_for_dist(score, geneset_name):
    # get mixtures and thresholds
    distributions = find_distribution(score, geneset_name)
    localizer_gmm = group_distributions(distributions, method="gmm")
    localizer_kmeans = group_distributions(distributions, method="kmeans")
    thresholds_gmm = find_grouped_dist_thresholds(
        distributions, localizer_gmm, score, geneset_name
    )
    thresholds_kmeans = find_grouped_dist_thresholds(
        distributions, localizer_kmeans, score, geneset_name
    )
    return (
        thresholds_gmm,
        thresholds_kmeans,
        distributions,
        localizer_gmm,
        localizer_kmeans,
    )


score_names = ["ft", "dino", "log2", "raw_data", "sctrans", "seurat"]  # data types

if __name__ == "__main__":
    data_type = "AUCell"
    print(data_type)
    for score_name in tqdm(score_names):
        print(score_name)
        # get scores
        scores = pd.read_csv(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell.csv", index_col=0
        )
        gs_names = scores.index.values.tolist()
        scores = scores.to_numpy()
        scores_thr = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
        scores_thr_kmeans = pd.DataFrame(0, index=gs_names, columns=np.arange(1))
        scores_dist = []
        gmm_thrs = {}
        kmeans_thrs = {}
        locs_gmm = []
        locs_kmeans = []
        for i, gs_name in tqdm(enumerate(gs_names), total=len(gs_names)):
            score = scores[i, :]
            (
                thresholds_gmm,
                thresholds_kmeans,
                distributions,
                localizer_gmm,
                localizer_kmeans,
            ) = pipeline_for_dist(score, gs_name)

            del distributions["TIC"], distributions["l_lik"]
            distributions["weights"] = (distributions["weights"]).tolist()
            distributions["mu"] = (distributions["mu"]).tolist()
            distributions["sigma"] = (distributions["sigma"]).tolist()

            if all(thresholds_gmm.shape):
                scores_thr.loc[gs_name] = thresholds_gmm[-1]
            else:
                scores_thr.loc[gs_name] = np.nan
            if all(thresholds_kmeans.shape):
                scores_thr_kmeans.loc[gs_name] = thresholds_kmeans[-1]
            else:
                scores_thr_kmeans.loc[gs_name] = np.nan

            scores_dist.append(distributions)

            gmm_thrs[gs_name] = thresholds_gmm.tolist()
            kmeans_thrs[gs_name] = thresholds_kmeans.tolist()

            localizer_gmm = localizer_gmm.tolist()
            locs_gmm.append(localizer_gmm)
            localizer_kmeans = localizer_kmeans.tolist()
            locs_kmeans.append(localizer_kmeans)

        scores_thr.to_csv(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_gmm_thr.csv"
        )
        scores_thr_kmeans.to_csv(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_kmeans_thr.csv"
        )

        with open(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_gmm_loc.json", "w"
        ) as fout:
            json.dump(locs_gmm, fout)
        with open(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_kmeans_loc.json",
            "w",
        ) as fout:
            json.dump(locs_kmeans, fout)

        with open(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_dist.json", "w"
        ) as fout:
            json.dump(scores_dist, fout)

        with open(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_gmm_thrs.json", "w"
        ) as fout:
            json.dump(gmm_thrs, fout)
        with open(
            "enrichment_stuff/data/AUCell/" + score_name + "/AUCell_kmeans_thrs.json",
            "w",
        ) as fout:
            json.dump(kmeans_thrs, fout)
